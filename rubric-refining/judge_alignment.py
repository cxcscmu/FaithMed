"""
judge_alignment.py — Score trajectories with gemini-2.5-flash-lite and compare
alignment against existing gemini-2.5-flash scores in latest_scores.json.

Usage:
    python judge_alignment.py [--force-rescore] [--workers N]

Outputs:
    scores/lite_scores.json   — flash-lite scores (same format as latest_scores.json)
    alignment_report.json     — per-rubric and overall alignment stats
    alignment_report.txt      — human-readable report
"""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent
sys.path.insert(0, str(_BASE))              # for pipeline, pruner
sys.path.insert(0, str(_BASE.parent / "rewards"))  # for rubrics_judge
import pipeline

from rubrics_judge import (  # prompt builder + helpers shared with RL training
    load_active_rubrics,
    _build_prompt,
    _check_precondition,
)
from google import genai
from google.genai import types

BASE_DIR         = Path(__file__).parent
LITE_MODEL       = "gemini-2.5-flash-lite"
# LITE_MODEL       = "gemini-3.1-flash-lite-preview"
LITE_CACHE_FILE  = BASE_DIR / "scores" / "lite_scores_cache_wo_reason_lite.json"
LITE_SCORES_FILE = BASE_DIR / "scores" / "lite_scores_wo_reason_lite.json"
ALIGNMENT_JSON   = BASE_DIR / "alignment_report_wo_reason_lite.json"
ALIGNMENT_TXT    = BASE_DIR / "alignment_report_wo_reason_lite.txt"
MAX_WORKERS      = 4

_cache_lock = threading.Lock()


# ── Cache ──────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    return json.loads(LITE_CACHE_FILE.read_text()) if LITE_CACHE_FILE.exists() else {}

def _save_cache(cache: dict) -> None:
    LITE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LITE_CACHE_FILE.write_text(json.dumps(cache, indent=2))

def _cache_key(traj_id: str, rubric_id: str, rubric_description: str) -> str:
    return hashlib.md5(f"{traj_id}::{rubric_id}::{rubric_description}".encode()).hexdigest()


# ── JSONL reader (reads trajectory files from disk) ───────────────────────────

def _read_jsonl_features(path: Path) -> dict:
    turns = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    think_blocks, search_queries, retrieved_passages, answers = [], [], [], []
    for turn in turns:
        resp = turn.get("response", "")
        think_blocks.extend(re.findall(r"<think>(.*?)</think>", resp, re.DOTALL))
        search_queries.extend(re.findall(r"<search>(.*?)</search>", resp, re.DOTALL))
        answers.extend(re.findall(r"<answer>(.*?)</answer>", resp, re.DOTALL))
        retrieved_passages.extend(
            re.findall(r"\[T\d+-R\d+\].*?(?=\[T\d+-R\d+\]|\Z)", turn.get("input", ""), re.DOTALL)
        )

    model_text = "\n\n".join(think_blocks + answers)
    cited = {(int(m.group(1)), int(m.group(2))) for m in re.finditer(r"T(\d+)-R(\d+)", model_text)}

    full_text = "\n\n".join(
        [f"[THINK {i+1}]\n{t}"          for i, t in enumerate(think_blocks)]
        + [f"[SEARCH {i+1}]\n{q}"       for i, q in enumerate(search_queries)]
        + [f"[RETRIEVED {i+1}]\n{p[:400]}" for i, p in enumerate(retrieved_passages)]
        + [f"[ANSWER]\n{a}"             for a in answers]
    )
    return {
        "search_performed": len(search_queries) > 0,
        "has_citations":    len(cited) > 0,
        "answer":           len(search_queries) > 0 and len(answers) > 0,
        "full_text":        full_text[:10000],
    }


# ── Gemini call (flash-lite, score-only, 512 tokens) ──────────────────────────

_gemini_client = None
_client_lock   = threading.Lock()

def _get_client():
    global _gemini_client
    if _gemini_client is None:
        with _client_lock:
            if _gemini_client is None:  # double-checked locking
                import os
                from dotenv import load_dotenv
                load_dotenv(os.environ["KEYS_ENV_PATH"], override=False)
                _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client

def _call_lite(prompt: str) -> str:
    config = types.GenerateContentConfig(
        max_output_tokens=512,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(5):
        try:
            resp = _get_client().models.generate_content(
                model=LITE_MODEL, contents=prompt, config=config
            )
            return resp.text.strip()
        except Exception as e:
            if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                raise
            wait = 10 * (2 ** attempt)
            print(f"    [503] overloaded, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"{LITE_MODEL} unavailable after 5 retries")


# ── Per-trajectory scorer ──────────────────────────────────────────────────────

def _score_trajectory(path: Path, rubrics: list[dict], cache: dict,
                      force_rescore: bool) -> dict[str, int]:
    traj_id  = path.stem
    features = _read_jsonl_features(path)
    result: dict[str, int] = {}
    to_score: list[dict]   = []

    for rubric in rubrics:
        rid = rubric["id"]
        if not _check_precondition(rubric, features):
            result[rid] = -1
            continue
        key = _cache_key(traj_id, rid, rubric["description"])
        if not force_rescore:
            with _cache_lock:
                cached = cache.get(key)
            if cached is not None:
                result[rid] = cached
                continue
        to_score.append(rubric)

    if not to_score:
        return result

    # Score-only prompt (include_reasoning=False) — matches rubrics_judge.py RL usage
    prompt = _build_prompt(to_score, features, include_reasoning=False)
    try:
        raw    = _call_lite(prompt)
        raw    = re.sub(r"^```json\s*", "", raw)
        raw    = re.sub(r"\s*```$",     "", raw)
        parsed = json.loads(raw)
        for rubric in to_score:
            rid   = rubric["id"]
            entry = parsed.get(rid)
            try:
                score = int(entry) if not isinstance(entry, dict) else int(entry["score"])
            except (TypeError, ValueError):
                score = -1
            with _cache_lock:
                cache[_cache_key(traj_id, rid, rubric["description"])] = score
            result[rid] = score
    except Exception as e:
        print(f"    [WARN] {traj_id} failed: {e}")
        for rubric in to_score:
            rid = rubric["id"]
            with _cache_lock:
                cache[_cache_key(traj_id, rid, rubric["description"])] = -1
            result[rid] = -1

    return result


# ── Alignment metrics ──────────────────────────────────────────────────────────

def cohen_kappa(y1: list[int], y2: list[int], weights: str | None = None) -> float:
    pairs = [(a, b) for a, b in zip(y1, y2) if a != -1 and b != -1]
    if len(pairs) < 2:
        return float("nan")
    a_vals, b_vals = zip(*pairs)
    labels    = sorted(set(a_vals) | set(b_vals))
    n         = len(labels)
    label_idx = {v: i for i, v in enumerate(labels)}

    cm = np.zeros((n, n), dtype=float)
    for a, b in pairs:
        cm[label_idx[a], label_idx[b]] += 1
    cm /= cm.sum()

    if weights == "linear" and n > 1:
        w  = np.array([[1 - abs(i - j) / (n - 1) for j in range(n)] for i in range(n)])
        po = (cm * w).sum()
        pe = (np.outer(cm.sum(axis=1), cm.sum(axis=0)) * w).sum()
    else:
        po = np.trace(cm)
        pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum()

    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def compute_alignment(flash_rows: list[dict], lite_scores: dict[str, dict[str, int]],
                      rubrics: list[dict]) -> dict:
    rubric_map   = {r["id"]: r for r in rubrics}
    flash_lookup = {row["trajectory"]: {k: v for k, v in row.items() if k != "trajectory"}
                    for row in flash_rows}

    per_rubric: dict[str, dict] = {}
    overall_flash, overall_lite = [], []

    for rid in [r["id"] for r in rubrics]:
        rubric = rubric_map[rid]
        pairs  = [(f_row.get(rid, -1), lite_scores[tid].get(rid, -1))
                  for tid, f_row in flash_lookup.items() if tid in lite_scores]
        pairs  = [(f, l) for f, l in pairs if not (f == -1 and l == -1)]
        n      = len(pairs)

        if n == 0:
            per_rubric[rid] = {"n": 0}
            continue

        f_scores, l_scores = zip(*pairs)
        exact  = sum(f == l for f, l in pairs) / n
        kappa  = cohen_kappa(list(f_scores), list(l_scores),
                             weights="linear" if rubric["type"] == "ordinal3" else None)
        valid  = [(f, l) for f, l in pairs if f != -1 and l != -1]
        mae    = float(np.mean([abs(f - l) for f, l in valid])) if valid else float("nan")

        per_rubric[rid] = {
            "name": rubric["name"], "type": rubric["type"], "n": n,
            "exact_agree": round(exact, 4),
            "kappa": round(float(kappa), 4) if not np.isnan(kappa) else None,
            "mae":   round(mae, 4)          if not np.isnan(mae)   else None,
        }
        if valid:
            f_vals, l_vals = zip(*valid)
            overall_flash.extend(f_vals)
            overall_lite.extend(l_vals)

    n_ov  = len(overall_flash)
    o_kappa = cohen_kappa(overall_flash, overall_lite)
    return {
        "overall": {
            "n_pairs":     n_ov,
            "exact_agree": round(sum(f == l for f, l in zip(overall_flash, overall_lite)) / n_ov, 4) if n_ov else 0.0,
            "kappa":       round(float(o_kappa), 4) if not np.isnan(o_kappa) else None,
            "mae":         round(float(np.mean([abs(f - l) for f, l in zip(overall_flash, overall_lite)])), 4) if n_ov else float("nan"),
        },
        "per_rubric": per_rubric,
    }


# ── Report printer ─────────────────────────────────────────────────────────────

def format_report(alignment: dict, flash_model: str, lite_model: str) -> str:
    lines = ["=" * 72, "  JUDGE ALIGNMENT REPORT",
             f"  Flash (reference): {flash_model}",
             f"  Lite  (candidate): {lite_model}", "=" * 72]
    o = alignment["overall"]
    lines += [f"\n  Overall ({o['n_pairs']} valid score pairs across all rubrics)",
              f"    Exact agreement : {o['exact_agree']:.1%}",
              f"    Cohen's κ       : {o['kappa']}",
              f"    Mean abs error  : {o['mae']:.4f}",
              f"\n  {'ID':<8} {'Type':<10} {'N':>5} {'Agree':>7} {'κ':>7} {'MAE':>6}  Name",
              "  " + "-" * 70]
    for rid, s in sorted(alignment["per_rubric"].items()):
        if s.get("n", 0) == 0:
            continue
        lines.append(
            f"  {rid:<8} {s['type']:<10} {s['n']:>5} {s['exact_agree']:>6.1%} "
            f"{s['kappa']:>7.3f}  {s['mae']:>5.3f}  {s['name']}"
            if s["kappa"] is not None else
            f"  {rid:<8} {s['type']:<10} {s['n']:>5} {s['exact_agree']:>6.1%}     N/A    N/A  {s['name']}"
        )
    lines += ["", "  κ: <0 poor · 0–0.2 slight · 0.2–0.4 fair · 0.4–0.6 moderate · 0.6–0.8 substantial · >0.8 almost perfect",
              "=" * 72]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    rubrics    = load_active_rubrics()
    flash_rows = json.loads(pipeline.SCORES_EXPORT.read_text())
    traj_ids   = {row["trajectory"] for row in flash_rows}

    traj_files = sorted(p for p in pipeline.DEFAULT_TRAJECTORY_DIR.glob("*.jsonl")
                        if p.stem in traj_ids)
    missing = traj_ids - {p.stem for p in traj_files}
    if missing:
        print(f"  [WARN] {len(missing)} trajectory files not found")

    print(f"Active rubrics : {len(rubrics)}")
    print(f"Trajectories   : {len(traj_files)}")
    print(f"Lite model     : {LITE_MODEL}")
    print(f"\nScoring {len(traj_files)} trajectories × {len(rubrics)} rubrics  [workers={args.workers}]...")

    cache     = _load_cache()
    lite_scores: dict[str, dict[str, int]] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(_score_trajectory, p, rubrics, cache, args.force_rescore): p
            for p in traj_files
        }
        for future in as_completed(future_to_path):
            traj_id = future_to_path[future].stem
            try:
                lite_scores[traj_id] = future.result()
            except Exception as e:
                print(f"  [WARN] {traj_id}: {e}")
                lite_scores[traj_id] = {r["id"]: -1 for r in rubrics}
            completed += 1
            if completed % 10 == 0 or completed == len(traj_files):
                with _cache_lock:
                    _save_cache(cache)
                print(f"  [{completed}/{len(traj_files)}] scored")

    _save_cache(cache)

    # Export lite scores
    lite_rows = [{"trajectory": tid, **{r["id"]: lite_scores[tid].get(r["id"], -1) for r in rubrics}}
                 for tid in sorted(lite_scores)]
    LITE_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    LITE_SCORES_FILE.write_text(json.dumps(lite_rows, indent=2))
    print(f"  Lite scores → {LITE_SCORES_FILE}  ({len(lite_rows)} trajectories)")

    alignment = compute_alignment(flash_rows, lite_scores, rubrics)
    ALIGNMENT_JSON.write_text(json.dumps(alignment, indent=2))

    report = format_report(alignment, flash_model="gemini-2.5-flash", lite_model=LITE_MODEL)
    ALIGNMENT_TXT.write_text(report)
    print(f"\n{report}")
    print(f"\n  Report → {ALIGNMENT_TXT}")
    print(f"  JSON   → {ALIGNMENT_JSON}")


if __name__ == "__main__":
    main()
