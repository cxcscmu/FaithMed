"""
analyzer.py — Statistical analysis of rubric scores.

Computes per-rubric statistics and identifies candidates for pruning.
"""

import numpy as np
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────
VAR_THRESHOLD     = 0.04   # variance < this → ZERO-VAR flag (std < 0.2)
NA_RATE_THRESHOLD = 0.80   # NA% > this → HIGH-NA flag
CORR_THRESHOLD    = 0.75   # |Pearson r| > this → merge candidate
MET_CEILING       = 0.95   # binary MET% > this → CEILING flag
MET_FLOOR         = 0.05   # binary MET% < this → FLOOR flag


@dataclass
class RubricStats:
    rubric_id: str
    n_total: int
    n_valid: int
    na_rate: float
    mean: Optional[float]
    std: Optional[float]
    variance: Optional[float]
    met_rate: Optional[float]
    is_binary: bool
    scores: list[int]

    flag_zero_variance: bool = False
    flag_high_na: bool = False
    flag_ceiling: bool = False
    flag_floor: bool = False

    def compute_flags(self):
        if self.variance is not None and self.variance < VAR_THRESHOLD:
            self.flag_zero_variance = True
        if self.na_rate > NA_RATE_THRESHOLD:
            self.flag_high_na = True
        if self.is_binary and self.met_rate is not None:
            if self.met_rate > MET_CEILING:
                self.flag_ceiling = True
            if self.met_rate < MET_FLOOR:
                self.flag_floor = True

    @property
    def any_flag(self) -> bool:
        return (self.flag_zero_variance or self.flag_high_na
                or self.flag_ceiling or self.flag_floor)

    def to_dict(self) -> dict:
        return {
            "n_valid":   self.n_valid,
            "na_rate":   round(self.na_rate, 4),
            "mean":      round(self.mean, 4)     if self.mean     is not None else None,
            "std":       round(self.std, 4)      if self.std      is not None else None,
            "variance":  round(self.variance, 4) if self.variance is not None else None,
            "met_rate":  round(self.met_rate, 4) if self.met_rate is not None else None,
            "flags": {
                "zero_variance": self.flag_zero_variance,
                "high_na":       self.flag_high_na,
                "ceiling":       self.flag_ceiling,
                "floor":         self.flag_floor,
            },
        }


@dataclass
class CorrelationFlag:
    rubric_a: str
    rubric_b: str
    pearson_r: float
    n_shared: int


def compute_rubric_stats(
    scores: dict[str, dict[str, int]],
    rubrics: list[dict],
) -> dict[str, RubricStats]:
    active_ids = {r["id"]: r for r in rubrics if r.get("active", True) and not r.get("needs_redefinition", False)}
    stats = {}

    for rid, rubric_def in active_ids.items():
        raw   = [scores[traj].get(rid, -1) for traj in scores]
        valid = [s for s in raw if s != -1]
        n_total, n_valid = len(raw), len(valid)
        na_rate = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0

        is_binary = rubric_def["type"] == "binary"
        if n_valid == 0:
            s = RubricStats(rid, n_total, 0, na_rate, None, None, None, None, is_binary, raw)
        else:
            arr = np.array(valid, dtype=float)
            s = RubricStats(
                rubric_id=rid, n_total=n_total, n_valid=n_valid, na_rate=na_rate,
                mean=float(arr.mean()), std=float(arr.std()), variance=float(arr.var()),
                met_rate=float(arr.mean()) if is_binary else None,
                is_binary=is_binary, scores=raw,
            )
        s.compute_flags()
        stats[rid] = s
    return stats


def find_correlated_pairs(
    scores: dict[str, dict[str, int]],
    rubrics: list[dict],
    threshold: float = CORR_THRESHOLD,
) -> list[CorrelationFlag]:
    active_ids = [r["id"] for r in rubrics if r.get("active", True) and not r.get("needs_redefinition", False)]
    flags = []

    for id_a, id_b in combinations(active_ids, 2):
        pairs = [
            (scores[t].get(id_a, -1), scores[t].get(id_b, -1))
            for t in scores
        ]
        valid = [(a, b) for a, b in pairs if a != -1 and b != -1]
        if len(valid) < 5:
            continue
        arr = np.array(valid, dtype=float)
        if arr[:, 0].std() < 1e-9 or arr[:, 1].std() < 1e-9:
            continue
        r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        if abs(r) >= threshold:
            flags.append(CorrelationFlag(id_a, id_b, r, len(valid)))

    return sorted(flags, key=lambda x: -abs(x.pearson_r))


def print_stats_table(stats: dict[str, RubricStats]) -> None:
    print(f"\n  {'ID':5s} {'n_valid':>7s} {'NA%':>6s} {'mean':>6s} {'std':>6s} {'MET%':>6s}  flags")
    print(f"  {'─'*60}")
    for rid, s in sorted(stats.items()):
        flags = []
        if s.flag_zero_variance: flags.append("ZERO-VAR")
        if s.flag_high_na:       flags.append("HIGH-NA")
        if s.flag_ceiling:       flags.append("CEILING")
        if s.flag_floor:         flags.append("FLOOR")
        flag_str = ", ".join(flags) if flags else "—"
        met_str  = f"{s.met_rate:.2f}" if s.met_rate  is not None else "  — "
        mean_str = f"{s.mean:.3f}"     if s.mean      is not None else "  — "
        std_str  = f"{s.std:.3f}"      if s.std       is not None else "  — "
        print(
            f"  {rid:5s} {s.n_valid:>7d} {s.na_rate:>6.1%} "
            f"{mean_str:>6s} {std_str:>6s} {met_str:>6s}  {flag_str}"
        )


def summarize_decisions(
    stats: dict[str, RubricStats],
    corr_flags: list[CorrelationFlag],
) -> list[dict]:
    decisions = []
    handled = set()

    # 1. Merge correlated pairs first
    for cf in corr_flags:
        if cf.rubric_a in handled or cf.rubric_b in handled:
            continue
        decisions.append({
            "action": "merge",
            "rubric_ids": [cf.rubric_a, cf.rubric_b],
            "reason": f"Pearson r={cf.pearson_r:.3f} (n={cf.n_shared}) ≥ threshold {CORR_THRESHOLD}",
        })
        handled.update([cf.rubric_a, cf.rubric_b])

    # 2. Remove zero-variance (not high-NA — those need redefinition not removal)
    for rid, s in stats.items():
        if rid in handled:
            continue
        if s.flag_zero_variance and not s.flag_high_na:
            decisions.append({
                "action": "remove",
                "rubric_ids": [rid],
                "reason": (
                    f"variance={s.variance:.4f} < {VAR_THRESHOLD} "
                    f"(std={s.std:.3f}): no RL gradient signal"
                ),
            })
            handled.add(rid)

    # 3. Zero-variance AND high-NA → remove (doubly useless)
    for rid, s in stats.items():
        if rid in handled:
            continue
        if s.flag_zero_variance and s.flag_high_na:
            decisions.append({
                "action": "remove",
                "rubric_ids": [rid],
                "reason": (
                    f"variance={s.variance:.4f} AND NA={s.na_rate:.1%}: "
                    f"zero signal + rarely applicable"
                ),
            })
            handled.add(rid)

    # 4. High-NA only → redefine (concept is valid, precondition too strict)
    for rid, s in stats.items():
        if rid in handled:
            continue
        if s.flag_high_na:
            decisions.append({
                "action": "redefine",
                "rubric_ids": [rid],
                "reason": (
                    f"NA rate={s.na_rate:.1%} > {NA_RATE_THRESHOLD:.0%}: "
                    f"precondition too restrictive"
                ),
            })
            handled.add(rid)

    # 5. Ceiling/floor → redefine
    for rid, s in stats.items():
        if rid in handled:
            continue
        if s.flag_ceiling:
            decisions.append({
                "action": "redefine",
                "rubric_ids": [rid],
                "reason": (
                    f"MET rate={s.met_rate:.1%} > {MET_CEILING:.0%}: "
                    f"ceiling effect for this model; tighten definition"
                ),
            })
            handled.add(rid)
        elif s.flag_floor:
            decisions.append({
                "action": "redefine",
                "rubric_ids": [rid],
                "reason": (
                    f"MET rate={s.met_rate:.1%} < {MET_FLOOR:.0%}: "
                    f"floor effect; definition misaligned with observable behavior"
                ),
            })
            handled.add(rid)

    return decisions
