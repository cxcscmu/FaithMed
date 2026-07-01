"""
rubrics_judge.py — LLM-as-Judge process reward for MedRM RL training.

For each GRPO rollout group (group_n trajectories for the same question):
  1. Extract trajectory features from env.memory_context (in-memory, no disk I/O).
  2. Score all active rubrics via Gemini-2.5-flash — one batched call per trajectory,
     all trajectories in parallel (ThreadPoolExecutor).
  3. For each rubric R, mean-centre scores within the applicable subset
     (trajectories where R is not N/A).
     When fewer than 2 trajectories have R applicable, centred score = 0.0 (no signal).
  4. process_reward[T] = mean of centred scores over all applicable rubrics.
  5. Caller adds:  final_reward = outcome_reward + lambda_weight * process_reward.

Rubric file and Gemini backend are identical to rubric-refining/scorer.py.
"""

import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv(os.environ["KEYS_ENV_PATH"], override=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL    = "gemini-2.5-flash-lite"
MAX_WORKERS     = 8     # max concurrent Gemini calls
LAMBDA_DEFAULT  = 0.05
_RUBRICS_FILE   = Path(__file__).parent.parent / "rubric-refining" / "rubrics.yaml"
_rubrics_cache: list[dict] | None = None

# Debug: print Gemini input/output for the first N trajectories scored.
# Controlled by PROCESS_REWARD_DEBUG env var (0 = silent).
# Read lazily on first use so MedRMEnv.__init__ has time to set the env var.
_DEBUG_MAX_TRAJ: int = 0   # set via configure_debug() from MedRMEnv.__init__


def configure_debug(debug_max_traj: int) -> None:
    """Called once from MedRMEnv.__init__ to propagate process_reward_debug config."""
    global _DEBUG_MAX_TRAJ
    _DEBUG_MAX_TRAJ = debug_max_traj


def configure_gemini_model(model: str) -> None:
    """Called once from MedRMEnv.__init__ to propagate gemini_model config."""
    global GEMINI_MODEL
    GEMINI_MODEL = model
_debug_traj_count = 0
_debug_traj_lock  = threading.Lock()
_debug_centering_done = False  # print group centering only once


def load_active_rubrics() -> list[dict]:
    """Load active rubrics from rubrics.yaml, cached after the first call."""
    global _rubrics_cache
    if _rubrics_cache is None:
        data = yaml.safe_load(_RUBRICS_FILE.read_text())
        _rubrics_cache = [
            r for r in data["rubrics"]
            if r.get("active", True) and not r.get("needs_redefinition", False)
        ]
    return _rubrics_cache

def extract_features(memory_context: list[str]) -> dict:
    """
    Extract trajectory features from env.memory_context.

    Each entry in memory_context is a string of the form:
        "[Turn N]:\\n{response}\\n{observation}\\n"

    Mirrors scorer.py's _extract_trajectory_features but operates on in-memory
    strings rather than JSONL files.
    """
    full_text = "\n".join(memory_context)

    think_blocks      = re.findall(r"<think>(.*?)</think>",   full_text, re.DOTALL)
    search_queries    = re.findall(r"<search>(.*?)</search>",  full_text, re.DOTALL)
    answers           = re.findall(r"<answer>(.*?)</answer>",  full_text, re.DOTALL)
    retrieved_passages = re.findall(
        r"\[T\d+-R\d+\].*?(?=\[T\d+-R\d+\]|\Z)", full_text, re.DOTALL
    )

    model_text = "\n\n".join(think_blocks + answers)
    cited = {
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"T(\d+)-R(\d+)", model_text)
    }

    combined = "\n\n".join(
        [f"[THINK {i+1}]\n{t}"        for i, t in enumerate(think_blocks)]
        + [f"[SEARCH {i+1}]\n{q}"     for i, q in enumerate(search_queries)]
        + [f"[RETRIEVED {i+1}]\n{p[:400]}" for i, p in enumerate(retrieved_passages)]
        + [f"[ANSWER]\n{a}"           for a in answers]
    )

    return {
        "search_performed": len(search_queries) > 0,
        "num_searches":     len(search_queries),
        "answer":           len(search_queries) > 0 and len(answers) > 0,
        "full_text":        combined[:10000],
    }


def _check_precondition(rubric: dict, features: dict) -> bool:
    pre = rubric.get("precondition")
    if pre is None:
        return True
    return bool(features.get(pre, False))


def _build_prompt(applicable_rubrics: list[dict], features: dict,
                  include_reasoning: bool = False) -> str:
    rubric_blocks = []
    for r in applicable_rubrics:
        scale = r["scale_labels"]
        if r["type"] == "binary":
            scale_desc = f"Score 1 (MET) or 0 (UNMET). MET={scale[1]}  UNMET={scale[0]}"
        else:
            scale_desc = f"Score 0/1/2. 0={scale[0]}  1={scale[1]}  2={scale[2]}"
        rubric_blocks.append(
            f"### {r['id']}: {r['name']}\n{r['description'].strip()}\n{scale_desc}"
        )

    rubrics_text = "\n\n".join(rubric_blocks)
    ids          = [r["id"] for r in applicable_rubrics]

    if include_reasoning:
        example_entry = '{"score": <number>, "reasoning": "<one sentence>"}'
        example_obj   = "{" + ", ".join(f'"{i}": {example_entry}' for i in ids[:2]) + ", ...}"
        output_spec   = (
            f"Respond with ONLY valid JSON — a single object keyed by rubric ID:\n{example_obj}\n\n"
            "Each value must have \"score\" (integer) and \"reasoning\" (one sentence citing specific evidence).\n"
            "If a rubric's precondition does not apply, score it -1 to indicate N/A."
        )
    else:
        example_obj = "{" + ", ".join(f'"{i}": <integer>' for i in ids[:3]) + ", ...}"
        output_spec  = (
            f"Respond with ONLY valid JSON — a flat object mapping rubric ID to integer score:\n{example_obj}\n\n"
            "If a rubric's precondition does not apply (e.g., requires search but no search was performed), "
            "use -1 to indicate N/A."
        )

    return f"""You are an expert evaluator of AI medical reasoning trajectories.

## Trajectory
{features['full_text']}

## Rubrics to evaluate
{rubrics_text}

## Task
Score this trajectory on EVERY rubric listed above.
{output_spec}
"""


_gemini_client = None   # google.genai.Client singleton


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _call_gemini(prompt: str, max_output_tokens: int = 512) -> str:
    from google.genai import types
    client = _get_gemini_client()
    config = types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=config
    )
    return response.text.strip()


def _score_one(features: dict, active_rubrics: list[dict]) -> tuple[dict[str, int], str | None]:
    """
    Score all applicable rubrics for one trajectory in a single Gemini call.

    Returns:
        (scores, debug_str) where scores is {rubric_id: score} (-1 = N/A)
        and debug_str is a formatted string to print from the main thread
        (or None if debug is disabled for this call).
    """
    result: dict[str, int] = {}
    to_score: list[dict]   = []

    for rubric in active_rubrics:
        rid = rubric["id"]
        if not _check_precondition(rubric, features):
            result[rid] = -1
            continue
        to_score.append(rubric)

    if not to_score:
        return result, None

    prompt = _build_prompt(to_score, features)
    # Debug calls need more tokens for reasoning text; normal calls are tiny.
    max_tokens = 512

    raw = "(call failed)"
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            raw    = _call_gemini(prompt, max_output_tokens=max_tokens)
            raw    = re.sub(r"^```json\s*", "", raw)
            raw    = re.sub(r"\s*```$",     "", raw)
            parsed = json.loads(raw)
            for rubric in to_score:
                rid   = rubric["id"]
                entry = parsed.get(rid)
                try:
                    # Support both flat {"R1": 1} and nested {"R1": {"score": 1}} formats.
                    score = int(entry) if not isinstance(entry, dict) else int(entry["score"])
                except (KeyError, TypeError, ValueError):
                    score = -1
                result[rid] = score
            break
        except Exception as e:
            if attempt < max_attempts:
                sleep_s = min(2 ** (attempt - 1), 4) + random.random()
                print(
                    f"[rubrics_judge] Gemini call failed on attempt "
                    f"{attempt}/{max_attempts}: {e}; retrying in {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue

            print(f"[rubrics_judge] Gemini call failed after {max_attempts} attempts: {e}", flush=True)
            for rubric in to_score:
                result[rubric["id"]] = -1

    if not all(r["id"] in result for r in to_score):
        for rubric in to_score:
            result.setdefault(rubric["id"], -1)

    return result, None


def score_trajectory(
    memory_context: list[str],
    rubrics: Optional[list[dict]] = None,
) -> tuple[dict[str, int], str | None]:
    """
    Step 1 of two-phase process reward: score one trajectory via a Gemini call.

    Slow (one Gemini API call).  Call as soon as a trajectory's episode ends
    (done=True) so that scoring overlaps with the ongoing rollout of other
    trajectories.

    Returns:
        (scores, debug_str) — debug_str is non-None for the first
        _DEBUG_MAX_TRAJ trajectories; print it from the main thread.
    """
    if rubrics is None:
        rubrics = load_active_rubrics()
    features = extract_features(memory_context)
    return _score_one(features, rubrics)


def step_rubric_contributions(
    raw_scores: dict[str, int],
    step_type: str,
    rubrics: list[dict],
    lambda_weight: float = LAMBDA_DEFAULT,
    rescale_search_answer_rubrics: float = 1.5,
) -> list[dict]:
    """
    Per-rubric contribution breakdown for a single step.

    Same scoring logic as distribute_rubric_scores_to_steps but returns one
    entry per rubric rather than one aggregated float per step.

    Returns:
        list of {"id", "name", "raw", "applicable", "contrib"} dicts.
    """
    def _norm_score(r) -> float:
        return max(raw_scores.get(r["id"], -1), 0) / (2 if r["type"] == "ordinal3" else 1)

    search_answer_rubrics = [r for r in rubrics if r.get("precondition") in ("search_performed", "answer")]
    denom      = len(search_answer_rubrics) or 1
    reset_denom = sum(1 for r in rubrics if r.get("precondition") is None) or 1

    rows = []
    for r in rubrics:
        precond = r.get("precondition")
        if step_type == "reset":
            applicable = precond is None
            contrib = lambda_weight * _norm_score(r) / reset_denom if applicable else 0.0
        elif step_type == "search":
            applicable = precond == "search_performed"
            contrib = lambda_weight * rescale_search_answer_rubrics * _norm_score(r) / denom if applicable else 0.0
        elif step_type == "answer":
            applicable = precond in ("search_performed", "answer")
            contrib = lambda_weight * rescale_search_answer_rubrics * _norm_score(r) / denom if applicable else 0.0
        else:  # invalid
            applicable = False
            contrib = 0.0
        rows.append({
            "id": r["id"], "name": r.get("name", r["id"]),
            "raw": raw_scores.get(r["id"], -1),
            "applicable": applicable, "contrib": contrib,
        })
    return rows

def distribute_rubric_scores_to_steps(
    raw_scores: dict[str, int],
    step_types: list[str],
    rubrics: list[dict],
    lambda_weight: float = LAMBDA_DEFAULT,
    rescale_search_answer_rubrics: float = 1.5,
) -> list[float]:
    """
    Distribute trajectory-level rubric scores to per-step rewards for GiGPO.
    Each step receives the sum of applicable rubric contributions.
    """
    return [
        sum(row["contrib"] for row in step_rubric_contributions(
            raw_scores, st, rubrics, lambda_weight, rescale_search_answer_rubrics
        ))
        for st in step_types
    ]

def aggregate_trajectory_rubric_score(
    raw_scores: dict[str, int],
    rubrics: list[dict],
    lambda_weight: float = LAMBDA_DEFAULT,
) -> float:
    """
    Aggregate all rubric scores for a trajectory into a single scalar without
    step-type filtering.  Used by GiGPO when step_scoring=False to assign the
    same uniform step_reward to every step of a trajectory (ablation mode).

    Returns lambda_weight * mean(normalised scores over applicable rubrics).
    A rubric is considered N/A when its raw score is -1 and is excluded.
    """
    total = 0.0
    count = 0
    for r in rubrics:
        score = raw_scores.get(r["id"], -1)
        if score == -1:
            continue
        max_score = 2 if r["type"] == "ordinal3" else 1
        total += score / max_score
        count += 1
    return lambda_weight * (total / count) if count > 0 else 0.0


def apply_group_centering(
    group_raw_scores: list[dict[str, int]],
    rubrics: Optional[list[dict]] = None,
    lambda_weight: float = LAMBDA_DEFAULT,
) -> list[float]:
    """
    Step 2 of two-phase process reward: mean-centre and aggregate.

    Fast (pure math, no API calls).  Call once all trajectories in a GRPO
    group have been scored by score_trajectory().

    Args:
        group_raw_scores: list of {rubric_id: score} dicts, one per trajectory.
                          score == -1 means N/A for that rubric.
        rubrics:          active rubrics; loaded from rubrics.yaml if None.
        lambda_weight:    multiplier applied to the final process reward.

    Returns:
        list[float] of length n, each ∈ [−λ, +λ], group mean ≈ +0.05λ.
    """
    if rubrics is None:
        rubrics = load_active_rubrics()

    n = len(group_raw_scores)
    if n == 0:
        return []

    # centred[i] maps rubric_id → float (None = N/A, excluded from mean)
    centred: list[dict[str, Optional[float]]] = [{} for _ in range(n)]

    for rubric in rubrics:
        rid = rubric["id"]
        max_score = 2 if rubric["type"] == "ordinal3" else 1

        applicable = [
            (i, group_raw_scores[i].get(rid, -1) / max_score)
            for i in range(n)
            if group_raw_scores[i].get(rid, -1) != -1
        ]

        if len(applicable) < 2:
            for i, _ in applicable:
                centred[i][rid] = 0.0
            continue

        scores_01 = [s for _, s in applicable]
        mu        = sum(scores_01) / len(scores_01)
        for i, s in applicable:
            centred[i][rid] = s - mu + 0.05   # ∈ [−1, +1], mean shifted to +0.05

    process_rewards: list[float] = []
    for i in range(n):
        vals = [v for v in centred[i].values() if v is not None]
        # pr   = (sum(vals) / len(vals)) if vals else 0.0
        pr   = sum(vals)
        process_rewards.append(lambda_weight * pr)

    global _debug_centering_done
    if not _debug_centering_done and _DEBUG_MAX_TRAJ > 0:
        _debug_centering_done = True
        rubric_ids = [r["id"] for r in rubrics] if rubrics else list(group_raw_scores[0].keys())
        max_score_map = {r["id"]: (2 if r["type"] == "ordinal3" else 1) for r in rubrics} if rubrics else {}
        per_rubric = {}
        for rid in rubric_ids:
            ms = max_score_map.get(rid, 1)
            raw = [group_raw_scores[i].get(rid, -1) for i in range(n)]
            normalised = [s / ms if s != -1 else None for s in raw]
            per_rubric[rid] = {
                "raw_scores":   raw,
                "normalised":   normalised,
                "centred":      [centred[i].get(rid) for i in range(n)],
            }
        debug_out = {
            "per_rubric":      per_rubric,
            "process_rewards": [round(r, 6) for r in process_rewards],
        }
        print(f"\n{'='*70}")
        print(f"[PR-DEBUG] Group Centering Result (group_n={n})")
        print(f"{'='*70}")
        print(json.dumps(debug_out, indent=2))

    return process_rewards

