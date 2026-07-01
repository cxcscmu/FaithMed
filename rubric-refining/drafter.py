"""
drafter.py — Auto-drafts replacement rubrics for empty dimensions.

Called by pipeline.py when check_dimensional_balance() finds a dimension
with zero active rubrics after a pruning iteration.

The LLM is given:
  1. The dimension's role in the EBM framework
  2. The removed rubrics and the statistical reasons they were pruned
  3. Sample trajectory excerpts to anchor the definition to real observable behavior
  4. Hard constraints: precondition must be broadly applicable, definition must
     reference specific trajectory elements (<think>, <search>, [Tn-Rn], <answer>)

Drafted rubrics are added to rubrics.yaml with auto_drafted: true so users can
identify and review them. They are scored normally in the next iteration — if
they also fail the statistical checks, they will be pruned or redefined again.
"""

import json
import re
import random
from pathlib import Path
from datetime import datetime

import yaml
from scorer import _call_llm

DIM_NAMES = {1: "ASK", 2: "ACQUIRE", 3: "APPRAISE", 4: "APPLY", 5: "ASSESS"}

DIM_ROLES = {
    1: (
        "ASK: The model must correctly formulate the clinical question before acting. "
        "This dimension captures whether the model understands *what* it needs to find out "
        "and *why*, prior to any search or answer. Observable in the Turn 1 <think> block."
    ),
    2: (
        "ACQUIRE: The model must translate identified knowledge gaps into effective search "
        "queries. This dimension captures the quality of <search> formulation and whether "
        "queries are atomic, targeted, and cover the necessary clinical components."
    ),
    3: (
        "APPRAISE: The model must critically engage with retrieved [Tn-Rn] passages — "
        "citing them, checking their accuracy, recognizing conflicts, and judging whether "
        "they are sufficient. Observable in the <think> blocks following each search turn."
    ),
    4: (
        "APPLY: The model must use retrieved evidence to build a grounded clinical argument — "
        "citing claims, eliminating incorrect options with evidence, and mapping evidence to "
        "the specific patient/scenario context in the question."
    ),
    5: (
        "ASSESS: The model must be epistemically calibrated — citations must correspond to "
        "real retrieved passages, cited passages must support their claims, and the model "
        "must hedge appropriately when evidence is weak or absent."
    ),
}

BROAD_PRECONDITIONS = ["null", "search_performed"]  # preconditions that cover ≥30% of trajectories


def _get_removed_rubrics(dimension: int, rubrics: list[dict]) -> list[dict]:
    """Return inactive rubrics from this dimension with their pruning history."""
    return [
        r for r in rubrics
        if r["dimension"] == dimension
        and not r.get("active", True)
        and r.get("pruning_history")
    ]


def _sample_trajectory_excerpts(trajectory_dir: Path, n: int = 3) -> list[str]:
    """Return short excerpts from n random trajectories."""
    files = list(trajectory_dir.glob("*.jsonl"))
    if not files:
        return []
    sample = random.sample(files, min(n, len(files)))
    excerpts = []
    for path in sample:
        turns = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))
        # Extract first think block and first search (if any)
        text_parts = []
        for turn in turns[:2]:
            resp = turn.get("response", "")
            think = re.findall(r"<think>(.*?)</think>", resp, re.DOTALL)
            search = re.findall(r"<search>(.*?)</search>", resp, re.DOTALL)
            if think:
                text_parts.append(f"<think>{think[0][:600].strip()}</think>")
            if search:
                text_parts.append(f"<search>{search[0].strip()}</search>")
            if len(text_parts) >= 2:
                break
        if text_parts:
            excerpts.append(f"--- {path.stem[:50]} ---\n" + "\n".join(text_parts))
    return excerpts


def _next_draft_id(dimension: int, rubrics: list[dict]) -> str:
    """Generate a unique draft ID like A_d1, A_d2, ..."""
    dim_letter = "ABCDE"[dimension - 1]
    existing = {r["id"] for r in rubrics}
    for i in range(1, 20):
        candidate = f"{dim_letter}_d{i}"
        if candidate not in existing:
            return candidate
    return f"{dim_letter}_d_auto_{datetime.now().strftime('%H%M%S')}"


def _build_draft_prompt(
    dimension: int,
    removed_rubrics: list[dict],
    trajectory_excerpts: list[str],
    existing_ids: set[str],
    new_id: str,
) -> str:
    dim_name = DIM_NAMES[dimension]
    dim_role = DIM_ROLES[dimension]

    removed_block = ""
    if removed_rubrics:
        lines = []
        for r in removed_rubrics:
            history = r.get("pruning_history", [{}])
            reason = history[-1].get("reason", "unknown") if history else "unknown"
            lines.append(
                f"  - [{r['id']}] {r.get('name','')}\n"
                f"    Removed because: {reason}\n"
                f"    Original definition (first 300 chars): "
                f"{str(r.get('description',''))[:300]}"
            )
        removed_block = "Previously removed rubrics from this dimension:\n" + "\n".join(lines)
    else:
        removed_block = "No previously removed rubrics in this dimension (dimension was never populated)."

    excerpts_block = "\n\n".join(trajectory_excerpts) if trajectory_excerpts else "(no excerpts available)"

    return f"""You are designing a rubric for LLM-as-judge evaluation of medical AI agent trajectories.

## Context
The trajectory format contains:
- `<think>...</think>` — the model's internal reasoning chain
- `<search>query</search>` — search queries issued to a retrieval system
- `[Tn-Rn]` — inline citations to retrieved passages (Turn n, Result n)
- `<answer>...</answer>` — the final answer with \\boxed{{choice}}

## Your task
Dimension {dimension} ({dim_name}) has no active rubrics. You must draft ONE replacement rubric.

## Dimension role
{dim_role}

## {removed_block}

## Why rubrics were removed (common patterns to AVOID):
- Zero variance: model always scores the same (e.g., always MET for simple behaviors)
- Ceiling effect: MET rate >95% — definition too easy to satisfy
- Floor effect: MET rate <5% — definition asks for behavior not observable in trajectories
- High NA rate: precondition met by <30% of trajectories

## Sample trajectory excerpts (to anchor your definition to real observable behavior)
{excerpts_block}

## Hard constraints for the new rubric
1. `precondition` MUST be one of: null (applies to all 70 trajectories) or "search_performed" (applies to ~43/70 trajectories). DO NOT use rare preconditions.
2. The `description` MUST reference at least one of: `<think>`, `<search>`, `[Tn-Rn]`, `<answer>` by name.
3. The rubric must capture a behavior that VARIES across trajectories — some models do it, some don't. Avoid behaviors that are trivially always done or never done.
4. The rubric must be DIFFERENT from behaviors already captured by rubrics in other dimensions.
5. Do NOT re-create any of the removed rubrics listed above — they were removed for statistical reasons.

## Output format
Respond with ONLY a valid YAML block (no markdown fences, no explanation).

CRITICAL formatting rules:
- `description` must contain the FULL scoring protocol: the inspection procedure, concrete MET criteria
  with a trajectory example, and concrete UNMET criteria with a trajectory example.
  Model it after this pattern:
    "Inspect the <think> block before the first <search>. Does the model contain an explicit
     sentence restating the core clinical question? MET: <think> contains a sentence such as
     'The question asks whether X causes Y'. UNMET: <think> moves directly from listing answer
     options to issuing <search> without articulating what the model is trying to find."
- `scale_labels` must be SHORT LABELS ONLY — do NOT put scoring criteria or explanations here.
  For binary: ["UNMET", "MET"]
  For ordinal3: ["0: short label", "1: short label", "2: short label"]

id: {new_id}
dimension: {dimension}
name: "short name (3-6 words)"
description: >
  [Inspection procedure + concrete MET example + concrete UNMET example.
   Must reference at least one of: <think>, <search>, [Tn-Rn], <answer>.]
type: binary  # or ordinal3
precondition: null  # or search_performed
scale_labels:
  - "UNMET"       # binary; or "0: short label" for ordinal3
  - "MET"         # binary; or "1: short label" / "2: short label" for ordinal3
active: true
auto_drafted: true
notes: "Auto-drafted to restore Dimension {dimension} ({dim_name}) coverage."
"""


def draft_replacement_rubric(
    dimension: int,
    rubrics: list[dict],
    trajectory_dir: Path,
    grader: str = "gemini",
) -> dict | None:
    """
    Draft one replacement rubric for an empty dimension.
    Returns a rubric dict ready to append to rubrics, or None on failure.
    """
    removed     = _get_removed_rubrics(dimension, rubrics)
    excerpts    = _sample_trajectory_excerpts(trajectory_dir)
    existing_ids = {r["id"] for r in rubrics}
    new_id      = _next_draft_id(dimension, rubrics)

    prompt = _build_draft_prompt(dimension, removed, excerpts, existing_ids, new_id)

    try:
        raw = _call_llm(prompt, grader=grader, max_output_tokens=4096, json_output=False, thinking=True)

        # Strip accidental markdown fences
        raw = re.sub(r"^```ya?ml\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        drafted = yaml.safe_load(raw)

        if not isinstance(drafted, dict):
            print(f"  [DRAFT WARN] LLM returned non-dict YAML ({type(drafted).__name__}) — skipping draft for Dim {dimension}")
            return None

        # Validate required fields
        required = {"id", "dimension", "name", "description", "type", "precondition", "scale_labels"}
        missing = required - set(drafted.keys())
        if missing:
            print(f"  [DRAFT WARN] Missing fields: {missing} — skipping draft for Dim {dimension}")
            return None

        # Enforce broad precondition
        if drafted.get("precondition") not in (None, "null", "search_performed"):
            print(f"  [DRAFT WARN] Precondition '{drafted['precondition']}' too restrictive — forcing null")
            drafted["precondition"] = None

        # Normalize null string
        if drafted.get("precondition") == "null":
            drafted["precondition"] = None

        drafted["active"] = True
        drafted["auto_drafted"] = True
        drafted.setdefault("notes", f"Auto-drafted to restore Dim {dimension} ({DIM_NAMES[dimension]}) coverage.")

        return drafted

    except Exception as e:
        print(f"  [DRAFT ERROR] Dim {dimension}: {e}")
        return None


def draft_redefined_rubric(
    rubric: dict,
    reason: str,
    rubrics: list[dict],
    trajectory_dir: Path,
    grader: str = "gemini",
) -> dict | None:
    """LLM-draft an improved replacement for a rubric flagged for redefinition."""
    dimension  = rubric["dimension"]
    dim_name   = DIM_NAMES.get(dimension, str(dimension))
    new_id     = _next_draft_id(dimension, rubrics)
    excerpts   = _sample_trajectory_excerpts(trajectory_dir)
    excerpts_block = "\n\n".join(excerpts) if excerpts else "(no excerpts available)"

    if "ceiling" in reason.lower():
        fix_hint = "CEILING: tighten the definition — require more specific or demanding behavior that fewer trajectories satisfy."
    elif "floor" in reason.lower():
        fix_hint = "FLOOR: anchor the definition to behavior that IS observable in the sample trajectories above."
    else:
        fix_hint = "HIGH-NA: broaden the precondition (use null or search_performed) so the rubric applies to more trajectories."

    prompt = f"""You are redesigning a rubric for LLM-as-judge evaluation of medical AI trajectories.

## Trajectory format
- `<think>...</think>` — model's internal reasoning
- `<search>query</search>` — retrieval queries
- `[Tn-Rn]` — citations to retrieved passages
- `<answer>...</answer>` — final answer

## Rubric to improve
id: {rubric['id']}
dimension: {dimension} ({dim_name})
name: {rubric['name']}
type: {rubric['type']}
precondition: {rubric.get('precondition') or 'null'}
description:
{rubric['description'].strip()}

## Why it was flagged
{reason}

## Fix required
{fix_hint}

## Sample trajectories
{excerpts_block}

## Hard constraints
1. id must be: {new_id}
2. dimension must be: {dimension}
3. precondition must be null or "search_performed"
4. Behavior must genuinely vary across trajectories (avoid trivially always-MET or always-UNMET)

## Format rules
- Make the MINIMAL change required to fix the flagged issue. Keep everything else identical.
- description: follow the exact same style as the original rubric above — same structure,
  same length, same use of inline examples. Do NOT expand or restructure it.
- scale_labels: copy the original scale_labels exactly unless the type changes.

Respond with ONLY valid YAML (no markdown fences):
id: {new_id}
dimension: {dimension}
name: "..."
description: >
  ...
type: {rubric['type']}
precondition: null  # or search_performed
scale_labels:
  - "..."
  - "..."
active: true
"""

    print(f"\n  [REDEFINE] [{rubric['id']}] {rubric['name']} — drafting improved version as [{new_id}]...")
    try:
        raw = _call_llm(prompt, grader=grader,
                        max_output_tokens=2048, json_output=False, thinking=False)
        raw = re.sub(r"^```ya?ml\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        drafted = yaml.safe_load(raw)

        if not isinstance(drafted, dict):
            print(f"  [REDEFINE WARN] LLM returned non-dict YAML ({type(drafted).__name__}) — keeping needs_redefinition flag for manual fix")
            return None

        required = {"id", "dimension", "name", "description", "type", "precondition", "scale_labels"}
        missing = required - set(drafted.keys())
        if missing:
            print(f"  [REDEFINE WARN] Missing fields: {missing} — keeping needs_redefinition flag for manual fix")
            return None

        drafted["id"] = new_id
        drafted["dimension"] = dimension
        if drafted.get("precondition") not in (None, "null", "search_performed"):
            drafted["precondition"] = None
        if drafted.get("precondition") == "null":
            drafted["precondition"] = None
        drafted["active"] = True
        drafted["auto_drafted"] = True
        drafted["notes"] = f"Auto-redefined from [{rubric['id']}]. Reason: {reason}"
        return drafted

    except Exception as e:
        print(f"  [REDEFINE ERROR] [{rubric['id']}]: {e}")
        return None


def draft_merged_rubric(
    rubric_a: dict,
    rubric_b: dict,
    reason: str,
    rubrics: list[dict],
    trajectory_dir: Path,
    grader: str = "gemini",
) -> dict | None:
    """LLM-draft a merged rubric combining two correlated rubrics."""
    dimension      = rubric_a["dimension"]
    new_id         = _next_draft_id(dimension, rubrics)
    excerpts       = _sample_trajectory_excerpts(trajectory_dir)
    excerpts_block = "\n\n".join(excerpts) if excerpts else "(no excerpts available)"

    prompt = f"""You are designing a merged rubric for LLM-as-judge evaluation of medical AI trajectories.

## Trajectory format
- `<think>...</think>` — model's internal reasoning
- `<search>query</search>` — retrieval queries
- `[Tn-Rn]` — citations to retrieved passages
- `<answer>...</answer>` — final answer

## Two correlated rubrics to merge
These rubrics are statistically redundant ({reason}) — combine them into one.

### [{rubric_a['id']}] {rubric_a['name']}
Type: {rubric_a['type']}  |  Precondition: {rubric_a.get('precondition') or 'null'}
{rubric_a['description'].strip()}

### [{rubric_b['id']}] {rubric_b['name']}
Type: {rubric_b['type']}  |  Precondition: {rubric_b.get('precondition') or 'null'}
{rubric_b['description'].strip()}

## Sample trajectories
{excerpts_block}

## Hard constraints
1. id must be: {new_id}
2. dimension must be: {dimension}
3. precondition must be null or "search_performed"
4. The merged behavior must still vary across trajectories

## Format rules
- description: follow the exact same style as the source rubrics above — same structure,
  same length, same use of inline examples. Do NOT invent a new format.
- Prefer the type of whichever source rubric has the richer scale (ordinal3 > binary),
  unless one type clearly fits better.
- scale_labels: follow the same style as the source rubrics (short labels only).

Respond with ONLY valid YAML (no markdown fences):
id: {new_id}
dimension: {dimension}
name: "..."
description: >
  ...
type: binary  # or ordinal3
precondition: null  # or search_performed
scale_labels:
  - "..."
  - "..."
active: true
"""

    print(f"\n  [MERGE] [{rubric_a['id']}] + [{rubric_b['id']}] — drafting merged rubric as [{new_id}]...")
    try:
        raw = _call_llm(prompt, grader=grader,
                        max_output_tokens=2048, json_output=False, thinking=False)
        raw = re.sub(r"^```ya?ml\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        drafted = yaml.safe_load(raw)

        if not isinstance(drafted, dict):
            print(f"  [MERGE WARN] LLM returned non-dict YAML ({type(drafted).__name__}) — merge failed")
            return None

        required = {"id", "dimension", "name", "description", "type", "precondition", "scale_labels"}
        missing = required - set(drafted.keys())
        if missing:
            print(f"  [MERGE WARN] Missing fields: {missing} — merge failed")
            return None

        drafted["id"] = new_id
        drafted["dimension"] = dimension
        if drafted.get("precondition") not in (None, "null", "search_performed"):
            drafted["precondition"] = None
        if drafted.get("precondition") == "null":
            drafted["precondition"] = None
        drafted["active"] = True
        drafted["auto_drafted"] = True
        drafted["notes"] = f"Auto-merged from [{rubric_a['id']}] + [{rubric_b['id']}]. {reason}"
        return drafted

    except Exception as e:
        print(f"  [MERGE ERROR] [{rubric_a['id']}]+[{rubric_b['id']}]: {e}")
        return None


def draft_replacements_for_empty_dims(
    empty_dims: list[int],
    rubrics: list[dict],
    trajectory_dir: Path,
    grader: str = "gemini",
) -> list[dict]:
    """
    Draft one replacement rubric per empty dimension.
    Returns list of drafted rubric dicts (may be shorter than empty_dims on failures).
    """
    drafted = []
    for dim in empty_dims:
        dim_name = DIM_NAMES.get(dim, str(dim))
        print(f"\n  [DRAFT] Dimension {dim} ({dim_name}) is empty — drafting replacement rubric...")
        rubric = draft_replacement_rubric(dim, rubrics + drafted, trajectory_dir, grader)
        if rubric:
            drafted.append(rubric)
            print(f"  [DRAFT] ✓ Drafted [{rubric['id']}]: {rubric['name']}")
            print(f"           Type: {rubric['type']}  |  Precondition: {rubric.get('precondition') or 'null (universal)'}")
            # Print first 200 chars of description
            desc = str(rubric.get("description", "")).strip().replace("\n", " ")
            print(f"           Desc: {desc[:200]}{'...' if len(desc) > 200 else ''}")
        else:
            print(f"  [DRAFT] ✗ Failed to draft rubric for Dim {dim} — manual intervention needed")
    return drafted
