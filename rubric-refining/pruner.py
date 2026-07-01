"""
pruner.py — Applies pruning decisions to the rubric set.

Supports: remove, merge, redefine (mark for human review), and
dimensional balance check (ensures every dimension has ≥1 active rubric).

After each iteration:
  - Saves a snapshot of rubrics.yaml to snapshots/rubrics_iter_NN.yaml
  - Writes a structured JSON log to logs/iteration_NN.json
  - Prints a human-readable before/after modification summary to stdout
"""

import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from copy import deepcopy

RUBRICS_FILE = Path(__file__).parent / "rubrics.yaml"
LOGS_DIR     = Path(__file__).parent / "logs"
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

DIM_NAMES = {1: "ASK", 2: "ACQUIRE", 3: "APPRAISE", 4: "APPLY", 5: "ASSESS"}


def load_rubrics(path: Path = RUBRICS_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text())["rubrics"]


def save_rubrics(rubrics: list[dict], path: Path = RUBRICS_FILE) -> None:
    path.write_text(yaml.dump({"rubrics": rubrics}, allow_unicode=True, sort_keys=False))


def _snapshot(iteration: int) -> None:
    """Save a per-iteration copy of rubrics.yaml to snapshots/."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = SNAPSHOTS_DIR / f"rubrics_iter_{iteration:02d}.yaml"
    shutil.copy(RUBRICS_FILE, dest)
    print(f"  [SNAPSHOT] {dest.name}")


def _log_iteration(iteration: int, decisions: list[dict], stats_summary: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "decisions": decisions,
        "stats_summary": stats_summary,
    }
    log_path = LOGS_DIR / f"iteration_{iteration:02d}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  [LOG]      {log_path.name}")


def _print_modification_summary(
    rubrics_before: list[dict],
    rubrics_after: list[dict],
    decisions: list[dict],
    stats: dict,
    iteration: int,
) -> None:
    """Print a human-readable before/after summary after each iteration."""
    SEP = "─" * 70

    active_before = {r["id"]: r for r in rubrics_before if r.get("active", True)}
    active_after  = {r["id"]: r for r in rubrics_after  if r.get("active", True)}

    removed  = [rid for rid in active_before if rid not in active_after]
    added    = [rid for rid in active_after  if rid not in active_before]

    print(f"\n{'═'*70}")
    print(f"  ITERATION {iteration} — MODIFICATION SUMMARY")
    print(f"{'═'*70}")
    print(f"  Active rubrics: {len(active_before)} → {len(active_after)}")
    print()

    # ── Per-decision details ──────────────────────────────────────────────
    if not decisions:
        print("  No changes made.")
    else:
        for d in decisions:
            action = d["action"].upper()
            ids    = d["rubric_ids"]
            reason = d["reason"]

            if action == "REMOVE":
                rid = ids[0]
                s = stats.get(rid)
                stat_str = ""
                if s:
                    if s.get("variance") is not None:
                        stat_str = f"var={s['variance']:.4f}, NA={s['na_rate']:.0%}"
                        if s.get("met_rate") is not None:
                            stat_str += f", MET={s['met_rate']:.0%}"
                print(f"  ✗ REMOVE  [{rid}] {_rubric_name(rubrics_before, rid)}")
                if stat_str:
                    print(f"            Stats: {stat_str}")
                print(f"            Reason: {reason}")

            elif action == "MERGE":
                id_a, id_b = ids[0], ids[1]
                s_a = stats.get(id_a, {})
                s_b = stats.get(id_b, {})
                print(f"  ⇒ MERGE   [{id_a}] {_rubric_name(rubrics_before, id_a)}")
                print(f"       +    [{id_b}] {_rubric_name(rubrics_before, id_b)}")
                r_val = reason.split("r=")[-1].split(" ")[0] if "r=" in reason else "?"
                print(f"            Pearson r={r_val}  →  marked for merged definition")

            elif action == "REDEFINE":
                rid = ids[0]
                s = stats.get(rid, {})
                stat_str = ""
                if s.get("met_rate") is not None:
                    stat_str = f"MET={s['met_rate']:.0%}"
                elif s.get("na_rate") is not None:
                    stat_str = f"NA={s['na_rate']:.0%}"
                print(f"  ⚠ REDEFINE[{rid}] {_rubric_name(rubrics_before, rid)}")
                if stat_str:
                    print(f"            Stats: {stat_str}")
                print(f"            Reason: {reason}")

            print()

    # ── Dimensional balance after changes ─────────────────────────────────
    print(f"  {'─'*66}")
    print(f"  Dimensional balance after iteration {iteration}:")
    dims: dict[int, list] = {}
    for r in rubrics_after:
        if r.get("active", True):
            dims.setdefault(r["dimension"], []).append(r["id"])
    all_dims = sorted({r["dimension"] for r in rubrics_after})
    for d in all_dims:
        ids_in_dim = dims.get(d, [])
        status = "✓" if ids_in_dim else "✗ EMPTY"
        label = DIM_NAMES.get(d, str(d))
        print(f"    Dim {d} ({label:10s}): {status}  {ids_in_dim}")

    print(f"{'═'*70}\n")


def _rubric_name(rubrics: list[dict], rid: str) -> str:
    for r in rubrics:
        if r["id"] == rid:
            return r.get("name", "")
    return ""


def apply_remove(rubrics: list[dict], rubric_id: str, reason: str) -> list[dict]:
    updated = deepcopy(rubrics)
    for r in updated:
        if r["id"] == rubric_id:
            r["active"] = False
            r.setdefault("pruning_history", []).append({
                "action": "remove",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
    return updated


def apply_merge(
    rubrics: list[dict],
    rubric_id_a: str,
    rubric_id_b: str,
    merged_definition: dict,
    reason: str,
) -> list[dict]:
    updated = deepcopy(rubrics)
    for r in updated:
        if r["id"] in (rubric_id_a, rubric_id_b):
            r["active"] = False
            r.setdefault("pruning_history", []).append({
                "action": "merged_into",
                "merged_into": merged_definition["id"],
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
    merged_definition["active"] = True
    merged_definition.setdefault(
        "notes", f"Merged from {rubric_id_a} + {rubric_id_b}. Reason: {reason}"
    )
    updated.append(merged_definition)
    return updated


def apply_redefine_flag(rubrics: list[dict], rubric_id: str, reason: str) -> list[dict]:
    updated = deepcopy(rubrics)
    for r in updated:
        if r["id"] == rubric_id:
            r["needs_redefinition"] = True
            r.setdefault("pruning_history", []).append({
                "action": "flagged_for_redefinition",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
    return updated


def check_dimensional_balance(rubrics: list[dict]) -> list[int]:
    all_dims = sorted({r["dimension"] for r in rubrics})
    active_by_dim = {}
    for r in rubrics:
        if r.get("active", True) and not r.get("needs_redefinition", False):
            active_by_dim.setdefault(r["dimension"], []).append(r["id"])
    empty_dims = [d for d in all_dims if d not in active_by_dim]
    return empty_dims


def apply_decisions(
    rubrics: list[dict],
    decisions: list[dict],
    iteration: int,
    stats_summary: dict,
    auto_merge_definitions: dict | None = None,
) -> tuple[list[dict], bool]:
    """
    Apply all pruning decisions. After applying:
      1. Saves rubrics.yaml snapshot to snapshots/rubrics_iter_NN.yaml
      2. Writes JSON log to logs/iteration_NN.json
      3. Prints before/after modification summary

    Returns: (updated_rubrics, changed: bool)
    """
    rubrics_before = deepcopy(rubrics)
    changed = False
    updated = deepcopy(rubrics)

    for decision in decisions:
        action = decision["action"]
        ids    = decision["rubric_ids"]
        reason = decision["reason"]

        if action == "remove" and len(ids) == 1:
            updated = apply_remove(updated, ids[0], reason)
            changed = True

        elif action == "merge" and len(ids) == 2:
            key     = f"{ids[0]}+{ids[1]}"
            rev_key = f"{ids[1]}+{ids[0]}"
            merged_def = (auto_merge_definitions or {}).get(key) or \
                         (auto_merge_definitions or {}).get(rev_key)
            if merged_def:
                updated = apply_merge(updated, ids[0], ids[1], merged_def, reason)
                changed = True
            else:
                # No merged definition yet — flag both for human review
                for rid in ids:
                    other = [x for x in ids if x != rid][0]
                    updated = apply_redefine_flag(
                        updated, rid,
                        f"Correlated with {other} (r>{0.75}); merge definition needed"
                    )
                changed = True

        elif action == "redefine" and len(ids) == 1:
            updated = apply_redefine_flag(updated, ids[0], reason)
            changed = True

    # ── Persist ───────────────────────────────────────────────────────────
    save_rubrics(updated)
    _snapshot(iteration)
    _log_iteration(iteration, decisions, stats_summary)

    # ── Print summary ─────────────────────────────────────────────────────
    _print_modification_summary(rubrics_before, updated, decisions, stats_summary, iteration)

    return updated, changed
