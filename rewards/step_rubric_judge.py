"""
step_rubric_judge.py — Per-step LLM-as-Judge process reward for GiGPO.

Scores each model step immediately after generation.  Input to Gemini is:
  - retrieved passages from all PREVIOUS turns (with turn labels)
  - the current step's raw model response (<think>...</think> + action tag)

Rubric applicability uses dimension-based rules (additive, not exclusive):
  dimension 1 → step_type == "reset"
  dimension 2 → current action is search
  dimension 3 → there are previously retrieved passages
  dimension 4 → there are previously retrieved passages
  dimension 5 → step_type == "answer"

Reuses _score_one from rubrics_judge.
"""

from pathlib import Path
from typing import Callable, Optional

import yaml

from rewards.rubrics_judge import _score_one

_STEP_RUBRICS_FILE = Path(__file__).parent.parent / "rubric-refining" / "rubrics_step.yaml"
_step_rubrics_cache: list[dict] | None = None


def load_step_rubrics() -> list[dict]:
    """Load active step rubrics from rubrics_step.yaml, cached after the first call."""
    global _step_rubrics_cache
    if _step_rubrics_cache is None:
        data = yaml.safe_load(_STEP_RUBRICS_FILE.read_text())
        _step_rubrics_cache = [
            r for r in data["rubrics"]
            if r.get("active", True) and not r.get("needs_redefinition", False)
        ]
    return _step_rubrics_cache


def parse_disabled_dims(value) -> set[int]:
    """Parse disabled rubric dimensions from comma-separated config values."""
    if value is None or value == "":
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = str(value).replace(";", ",").split(",")

    dims: set[int] = set()
    for part in parts:
        text = str(part).strip()
        if text:
            dims.add(int(text))
    return dims


def filter_step_rubrics_by_disabled_dims(rubrics: list[dict], disabled_dims=None) -> list[dict]:
    """Keep all rubrics except those whose dimension is disabled for ablation."""
    dims = parse_disabled_dims(disabled_dims)
    if not dims:
        return rubrics
    return [r for r in rubrics if int(r.get("dimension", -1)) not in dims]


def _applicable_dims(
    step_type: str,
    prev_retrieved_passages: list[str],
    action: str | None,
) -> set[int]:
    """Return the set of rubric dimensions applicable to this step (rules are additive)."""
    dims: set[int] = set()
    if step_type == "reset":
        dims.add(1)
    if action and "<search>" in action:
        dims.add(2)
    if len(prev_retrieved_passages) > 0:
        dims.update({3, 4})
    if step_type == "answer":
        dims.add(5)
    return dims


def extract_step_features(
    original_response: str,
    prev_retrieved_passages: list[str],
) -> dict:
    """
    Build a features dict for a single step evaluation.

    Args:
        original_response:       Full raw model output, e.g.:
                                   "<think>...</think><search>query</search>"
                                 Already contains the think block + action tag.
        prev_retrieved_passages: Raw search results from all PREVIOUS turns,
                                 NOT including the current turn's search result.
    """
    retrieved_parts = [
        f"[Turn {i+1} Retrieved Information]\n{p[:400]}"
        for i, p in enumerate(prev_retrieved_passages)
    ]
    return {"full_text": "\n\n".join(retrieved_parts + [original_response])}


def step_score_contributions(
    scores: dict[str, int],
    rubrics: list[dict],
    lambda_weight: float = 0.05,
) -> list[dict]:
    """
    Per-rubric contribution breakdown for step_scoring mode (for logging).

    Returns one dict per applicable (non-N/A) rubric:
      {"id", "name", "dimension", "raw", "contrib"}
    where contrib = λ × normalized_score / count_applicable_in_same_dim.
    """
    dim_counts: dict[int, int] = {}
    for r in rubrics:
        d = r.get("dimension")
        if d is not None:
            dim_counts[d] = dim_counts.get(d, 0) + 1

    return [
        {
            "id":        r["id"],
            "name":      r.get("name", r["id"]),
            "dimension": r.get("dimension"),
            "raw":       scores[r["id"]],
            "contrib":   (
                lambda_weight
                * max(scores[r["id"]], 0)
                / (2 if r["type"] == "ordinal3" else 1)
                / dim_counts.get(r.get("dimension"), 1)
            ),
        }
        for r in rubrics
        if scores.get(r["id"], -1) != -1
    ]


def aggregate_step_scores(
    scores: dict[str, int],
    rubrics: list[dict],
    lambda_weight: float = 0.05,
) -> float:
    """
    Convert per-step Gemini scores to a single reward value.

    Strategy: for each dimension, take the mean of normalized applicable scores
    (score != -1), then sum across dimensions.  Each dimension contributes at
    most 1.0 × lambda_weight regardless of how many rubrics it contains.

        reward = λ × Σ_dim  mean({ s_norm : rubric in dim, s != -1 })

    No rescale factor needed — dimensions are already equal-weighted by the mean.
    """
    def _norm(r: dict) -> float:
        return max(scores.get(r["id"], -1), 0) / (2 if r["type"] == "ordinal3" else 1)

    # Group rubrics by dimension; skip dimensions where all scores are N/A
    dim_groups: dict[int, list[dict]] = {}
    for r in rubrics:
        d = r.get("dimension")
        if d is not None:
            dim_groups.setdefault(d, []).append(r)

    total = 0.0
    for dim_rubrics in dim_groups.values():
        applicable = [_norm(r) for r in dim_rubrics if scores.get(r["id"], -1) != -1]
        if applicable:
            total += sum(applicable) / len(applicable)

    return lambda_weight * total


def score_step(
    original_response: str,
    action: str | None,
    prev_retrieved_passages: list[str],
    step_type: str,
    rubrics: Optional[list[dict]] = None,
    features: Optional[dict] = None,
    score_one_fn: Optional[Callable[[dict, list[dict]], tuple[dict[str, int], str | None]]] = None,
) -> tuple[dict[str, int], str | None]:
    """
    Score a single model step immediately after it is generated.

    Applicable rubrics are selected by dimension (additive rules, see module docstring).
    invalid steps and steps with no applicable dimensions return all -1 without a Gemini call.

    Args:
        original_response:       Full raw model output for this step.
        action:                  Pre-parsed by projection.py: "<search>...</search>",
                                 "<answer>...</answer>", or None.
        prev_retrieved_passages: Retrieved passages from all PREVIOUS turns.
        step_type:               "reset", "search", "answer", or "invalid".
        rubrics:                 Pre-loaded step rubrics; loaded from rubrics_step.yaml if None.
        score_one_fn:            Optional scorer with the same contract as
                                 rewards.rubrics_judge._score_one. Defaults to Gemini.

    Returns:
        (scores_dict, debug_str) — same contract as score_trajectory().
        Non-applicable rubrics carry score -1 (N/A).
    """
    if rubrics is None:
        rubrics = load_step_rubrics()

    if step_type == "invalid":
        return {r["id"]: -1 for r in rubrics}, None

    dims = _applicable_dims(step_type, prev_retrieved_passages, action)
    applicable = [r for r in rubrics if r.get("dimension") in dims]
    if not applicable:
        return {r["id"]: -1 for r in rubrics}, None

    if features is None:
        features = extract_step_features(original_response, prev_retrieved_passages)
    if score_one_fn is None:
        score_one_fn = _score_one
    scores, debug_str = score_one_fn(features, applicable)
    return {r["id"]: scores.get(r["id"], -1) for r in rubrics}, debug_str
