"""
evaluation.py — Per-trajectory scores normalized to [0, 1], under two groupings.

For each trajectory and each group:
  - Only active rubrics in the group are considered.
  - Scores of -1 (not applicable) are excluded from both numerator and denominator.
  - binary rubric max = 1, ordinal3 rubric max = 2.
  - score = sum(valid scores) / sum(max scores for valid rubrics)

Usage:
    python evaluation.py
    python evaluation.py --scores path/to/latest_scores.json \
                         --rubrics path/to/rubrics.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

# ── Default paths ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
DEFAULT_SCORES  = _HERE / "qwen_scores" / "wo_rubrics_scores.json"
DEFAULT_RUBRICS = _HERE / "rubrics.yaml"

# ── Grouping 1: original 5-step EBM pipeline ─────────────────────────────────
DIMENSION_GROUPS: dict[str, list[str]] = {
    "Dim1_ASK":      ["A1", "A4", "A5"],
    "Dim2_ACQUIRE":  ["B5", "B6", "B_d2"],
    "Dim3_APPRAISE": ["C1", "C2", "C3", "C4", "C7"],
    "Dim4_APPLY":    ["D3", "D6", "D_d3"],
    "Dim5_ASSESS":   ["E2", "E4", "E6"],
}

# ── Grouping 2: thematic ──────────────────────────────────────────────────────
THEME_GROUPS: dict[str, list[str]] = {
    "Clinical_Reasoning_Completeness": ["A1", "A4", "A5", "B_d2", "D3", "E4"],
    "Search_Citation_Faithfulness":    ["B5", "B6", "C1", "D_d3"],
    "Factual_Accuracy":                ["C2", "C3", "D6"],
    "Safety_Limitations_Epistemic":    ["C4", "C7", "E2", "E6"],
}

MAX_SCORE = {"binary": 1, "ordinal3": 2}


def load_rubric_meta(rubrics_path: Path) -> dict[str, dict]:
    """Return {rubric_id: {type, active}} for all rubrics."""
    with open(rubrics_path) as f:
        data = yaml.safe_load(f)
    rubrics = data["rubrics"] if isinstance(data, dict) and "rubrics" in data else data
    return {
        r["id"]: {"type": r["type"], "active": r.get("active", True)}
        for r in rubrics
    }


def compute_dim_score(
    traj_scores: dict[str, int],
    rubric_ids: list[str],
    meta: dict[str, dict],
) -> float | None:
    """
    Compute normalized [0,1] score for one dimension on one trajectory.
    Returns None if no rubric in the group was applicable (all -1 or inactive).
    """
    total_score = 0
    total_max   = 0
    for rid in rubric_ids:
        if rid not in meta or not meta[rid]["active"]:
            continue
        score = traj_scores.get(rid, -1)
        if score == -1:
            continue
        rtype = meta[rid]["type"]
        total_score += score
        total_max   += MAX_SCORE.get(rtype, 1)

    if total_max == 0:
        return None
    return total_score / total_max


def evaluate(
    scores_path: Path,
    rubrics_path: Path,
    groups: dict[str, list[str]],
    output_path: Path,
) -> list[dict]:
    with open(scores_path) as f:
        all_scores: list[dict] = json.load(f)

    meta = load_rubric_meta(rubrics_path)

    results = []
    for entry in all_scores:
        traj_id = entry["trajectory"]
        row: dict = {"trajectory": traj_id}
        for group_name, rubric_ids in groups.items():
            score = compute_dim_score(entry, rubric_ids, meta)
            row[group_name] = round(score, 4) if score is not None else None
        results.append(row)

    group_names = list(groups.keys())
    summary: dict = {"trajectory": "all"}
    for g in group_names:
        vals = [r[g] for r in results if r[g] is not None]
        summary[g] = round(sum(vals) / len(vals), 4) if vals else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([summary] + results, f, indent=2)
    print(f"Saved {len(results)} trajectory scores → {output_path}")

    _print_summary(results, groups)
    return results


def _print_summary(results: list[dict], groups: dict[str, list[str]]) -> None:
    group_names = list(groups.keys())
    col = 32
    print(f"\n{'Trajectory':<55}", end="")
    for g in group_names:
        print(f"  {g[:col]:>{col}}", end="")
    print()
    print("─" * (55 + (col + 2) * len(group_names)))

    for row in results:
        print(f"{row['trajectory']:<55}", end="")
        for g in group_names:
            v = row[g]
            print(f"  {f'{v:.3f}':>{col}}" if v is not None else f"  {'—':>{col}}", end="")
        print()

    print("\n" + "─" * (55 + (col + 2) * len(group_names)))
    for label, agg_fn in [("mean", lambda xs: sum(xs) / len(xs)),
                           ("min",  min),
                           ("max",  max)]:
        print(f"{label:<55}", end="")
        for g in group_names:
            vals = [r[g] for r in results if r[g] is not None]
            v = agg_fn(vals) if vals else None
            print(f"  {f'{v:.3f}':>{col}}" if v is not None else f"  {'—':>{col}}", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-group normalized scores")
    parser.add_argument("--scores",  type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS)
    args = parser.parse_args()

    scores_dir = args.scores.parent

    print("\n" + "=" * 60)
    print("  GROUPING 1: 5-step EBM pipeline")
    print("=" * 60)
    evaluate(args.scores, args.rubrics, DIMENSION_GROUPS,
             scores_dir / "wo_rubrics_dimension_step_scores.json")

    print("\n" + "=" * 60)
    print("  GROUPING 2: Thematic")
    print("=" * 60)
    evaluate(args.scores, args.rubrics, THEME_GROUPS,
             scores_dir / "wo_rubrics_theme_scores.json")
