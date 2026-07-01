"""
pipeline.py — Iterative rubric pruning pipeline.

Each iteration:
  1. Scores all active rubrics on all trajectories (LLM-as-judge, cached)
  2. Computes statistics (variance, NA rate, MET%, pairwise Pearson r)
  3. Identifies pruning candidates
  4. Applies decisions and prints a before/after modification summary
  5. Saves rubrics.yaml + a per-iteration snapshot to snapshots/rubrics_iter_NN.yaml
  6. Writes a JSON log to logs/iteration_NN.json
  7. Checks dimensional balance
  Repeats until no candidates found (CONVERGED).

Usage:
  python pipeline.py [--trajectory-dir PATH] [--max-iters N] [--force-rescore]
                     [--grader {gemini,claude}] [--disable-refinement]
"""

import argparse
import json
from pathlib import Path

from scorer   import score_all
from analyzer import compute_rubric_stats, find_correlated_pairs, print_stats_table, summarize_decisions
from pruner   import load_rubrics, save_rubrics, apply_decisions, check_dimensional_balance, RUBRICS_FILE, _snapshot
from drafter  import draft_replacements_for_empty_dims, draft_merged_rubric, draft_redefined_rubric

# DEFAULT_TRAJECTORY_DIR = Path(
#     "/data/group_data/cx_group/medrm/output/MedAgent"
#     "/deepseek_r1_medmix_ebm/logs_medcorp"
# )
# SCORES_EXPORT = Path(__file__).parent / "scores" / "latest_scores.json"

DEFAULT_TRAJECTORY_DIR = Path("/data/user_data/zhiyunz/evaluation/qwen3_1.7b_sft6964878/logs_medcorp")
SCORES_EXPORT = Path(__file__).parent / "qwen_scores" / "sft_base_scores.json"

# ── Provide merged rubric definitions here when the pipeline flags a merge ────
# Key format: "LOWER_ID+HIGHER_ID"  (alphabetical)
AUTO_MERGE_DEFINITIONS: dict[str, dict] = {}


def export_scores(scores: dict, rubrics: list[dict]) -> None:
    SCORES_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    active_ids = [r["id"] for r in rubrics
                  if r.get("active", True) and not r.get("needs_redefinition", False)]
    rows = [
        {"trajectory": tid, **{rid: s.get(rid, -1) for rid in active_ids}}
        for tid, s in scores.items()
    ]
    SCORES_EXPORT.write_text(json.dumps(rows, indent=2))
    print(f"  [SCORES]   latest_scores.json updated ({len(rows)} trajectories)")


def run_pipeline(
    trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
    max_iters: int = 10,
    force_rescore: bool = False,
    auto_draft: bool = True,
    grader: str = "gemini",
    max_trajectories: int | None = None,
    disable_refinement: bool = False,
    trajectory_filter: Path | None = None,
) -> None:
    print("=" * 70)
    print("  RUBRIC PRUNING PIPELINE")
    if disable_refinement:
        print("  [MODE] Scoring only — refinement disabled")
    print("=" * 70)

    trajectory_ids: set[str] | None = None
    if trajectory_filter is not None:
        rows = json.loads(trajectory_filter.read_text())
        trajectory_ids = {row["trajectory"] for row in rows}
        print(f"  [FILTER] {len(trajectory_ids)} trajectories from {trajectory_filter.name}")

    rubrics   = load_rubrics()
    iteration = 0

    while iteration < max_iters:
        iteration += 1
        active = [r for r in rubrics
                  if r.get("active", True) and not r.get("needs_redefinition", False)]
        print(f"\n{'─'*70}")
        print(f"  ITERATION {iteration}  |  Active rubrics: {len(active)}")
        print(f"{'─'*70}")

        if not active:
            print("  No active rubrics remaining. Stopping.")
            break

        # ── Score ──────────────────────────────────────────────────────────
        scores = score_all(
            trajectory_dir=trajectory_dir,
            rubrics=rubrics,
            force_rescore=(force_rescore and iteration == 1),
            grader=grader,
            max_trajectories=max_trajectories,
            trajectory_ids=trajectory_ids,
        )
        export_scores(scores, rubrics)

        # ── Analyze ────────────────────────────────────────────────────────
        stats      = compute_rubric_stats(scores, rubrics)
        corr_flags = find_correlated_pairs(scores, rubrics)

        print("\n  Statistics:")
        print_stats_table(stats)

        if corr_flags:
            print("\n  Correlated pairs (|r| ≥ threshold):")
            for cf in corr_flags:
                print(f"    {cf.rubric_a} ↔ {cf.rubric_b}  r={cf.pearson_r:.3f}  (n={cf.n_shared})")

        if disable_refinement:
            break

        # ── Decisions ──────────────────────────────────────────────────────
        decisions = summarize_decisions(stats, corr_flags)

        if not decisions:
            print("\n  ✓ No pruning candidates. Pipeline CONVERGED.")
            # Still snapshot the final state
            from pruner import _log_iteration
            _snapshot(iteration)
            _log_iteration(iteration, [], {rid: s.to_dict() for rid, s in stats.items()})
            break

        # ── Auto-draft merge definitions (LLM) ────────────────────────────
        # Manual definitions in AUTO_MERGE_DEFINITIONS take priority; LLM fills the rest.
        local_merge_defs = dict(AUTO_MERGE_DEFINITIONS)
        drafted_merges: list[dict] = []  # accumulate to avoid ID collisions across merges
        for d in decisions:
            if d["action"] != "merge":
                continue
            id_a, id_b = d["rubric_ids"]
            key, rev   = f"{id_a}+{id_b}", f"{id_b}+{id_a}"
            if key in local_merge_defs or rev in local_merge_defs:
                continue
            ra = next((r for r in rubrics if r["id"] == id_a), None)
            rb = next((r for r in rubrics if r["id"] == id_b), None)
            if ra and rb:
                merged = draft_merged_rubric(
                    ra, rb, d["reason"], rubrics + drafted_merges, trajectory_dir, grader
                )
                if merged:
                    local_merge_defs[key] = merged
                    drafted_merges.append(merged)
                    print(f"  [MERGE] ✓ Drafted [{merged['id']}]: {merged['name']}")
                else:
                    print(f"  [MERGE] ✗ Auto-draft failed for [{id_a}]+[{id_b}] — both flagged needs_redefinition")

        # Track which rubrics are about to be flagged for redefine
        redefine_targets = {
            d["rubric_ids"][0]: d["reason"]
            for d in decisions if d["action"] == "redefine"
        }

        # ── Apply (prints modification summary internally) ─────────────────
        rubrics, changed = apply_decisions(
            rubrics=rubrics,
            decisions=decisions,
            iteration=iteration,
            stats_summary={rid: s.to_dict() for rid, s in stats.items()},
            auto_merge_definitions=local_merge_defs,
        )

        # ── Auto-redefine: LLM replaces flagged rubrics immediately ───────
        if redefine_targets:
            auto_redefined = []
            for rid, reason in redefine_targets.items():
                original = next((r for r in rubrics if r["id"] == rid), None)
                if original is None:
                    continue
                new_rubric = draft_redefined_rubric(
                    original, reason, rubrics + auto_redefined, trajectory_dir, grader
                )
                if new_rubric:
                    # Fully deactivate original — LLM replacement takes over
                    for r in rubrics:
                        if r["id"] == rid:
                            r["active"] = False
                            r["needs_redefinition"] = False
                    auto_redefined.append(new_rubric)
                    print(f"  [REDEFINE] ✓ [{rid}] replaced by [{new_rubric['id']}]: {new_rubric['name']}")
                else:
                    print(f"  [REDEFINE] ✗ Auto-draft failed for [{rid}] — kept as needs_redefinition=True for manual fix")
            if auto_redefined:
                rubrics.extend(auto_redefined)
                save_rubrics(rubrics)
                _snapshot(iteration)   # overwrite snapshot with final state including new rubrics
                changed = True

        # ── Dimensional balance ────────────────────────────────────────────
        empty_dims = check_dimensional_balance(rubrics)
        if empty_dims and auto_draft:
            drafted = draft_replacements_for_empty_dims(
                empty_dims, rubrics, trajectory_dir, grader
            )
            if drafted:
                rubrics.extend(drafted)
                save_rubrics(rubrics)
                _snapshot(iteration)  # overwrite snapshot to include newly drafted rubrics
                changed = True  # ensure next iteration scores the new rubrics
        elif empty_dims:
            dim_names = {1:"ASK", 2:"ACQUIRE", 3:"APPRAISE", 4:"APPLY", 5:"ASSESS"}
            for d in empty_dims:
                print(
                    f"  ⚠ WARNING: Dimension {d} ({dim_names.get(d,'?')}) has no active rubrics. "
                    f"Add a replacement rubric to rubrics.yaml before next iteration."
                )

        if not changed:
            print("  ✓ No changes applied. Pipeline CONVERGED.")
            break

    else:
        print(f"\n  [WARN] Reached max iterations ({max_iters}) without convergence.")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RUBRIC SET")
    print("=" * 70)
    active_final  = [r for r in rubrics if r.get("active", True)]
    needs_review  = [r for r in rubrics if r.get("needs_redefinition", False)]
    dim_names = {1:"ASK", 2:"ACQUIRE", 3:"APPRAISE", 4:"APPLY", 5:"ASSESS"}

    dims: dict[int, list] = {}
    for r in active_final:
        dims.setdefault(r["dimension"], []).append(r["id"])
    for d, ids in sorted(dims.items()):
        print(f"  Dim {d} ({dim_names.get(d,'?'):10s}): {ids}")

    print(f"\n  Total active       : {len(active_final)}")
    if needs_review:
        print(f"  Needs redefinition : {[r['id'] for r in needs_review]}")
    print(f"\n  Rubrics file : {RUBRICS_FILE}")
    print(f"  Snapshots    : {Path(__file__).parent / 'snapshots'}")
    print(f"  Logs         : {Path(__file__).parent / 'logs'}")
    print(f"  Score cache  : {Path(__file__).parent / 'scores' / 'scores_cache.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative rubric pruning pipeline")
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--max-iters",      type=int,  default=10)
    parser.add_argument("--force-rescore",  action="store_true",
                        help="Ignore score cache and re-score everything")
    parser.add_argument("--no-auto-draft",  action="store_true",
                        help="Warn only when a dimension empties; do not auto-draft replacements")
    parser.add_argument("--grader", choices=["gemini", "claude"], default="gemini",
                        help="Grader model for scoring (default: gemini)")
    parser.add_argument("--disable-refinement", action="store_true",
                        help="Score and export only — skip decisions, merges, and rubric edits")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="Randomly sample this many trajectories per iteration (seed=42); use all if omitted")
    parser.add_argument("--trajectory-filter", type=Path, default=None,
                        help="Path to a scores JSON; only score trajectories whose IDs appear in that file")
    args = parser.parse_args()
    run_pipeline(
        args.trajectory_dir,
        args.max_iters,
        args.force_rescore,
        auto_draft=not args.no_auto_draft,
        grader=args.grader,
        max_trajectories=args.max_trajectories,
        disable_refinement=args.disable_refinement,
        trajectory_filter=args.trajectory_filter,
    )
