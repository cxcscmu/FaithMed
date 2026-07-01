"""
scorer.py — LLM-as-judge rubric scorer for MedAgent trajectories.

Scores all active rubrics for a trajectory in ONE batched LLM call, then
parallelises across trajectories with a ThreadPoolExecutor.

Return values per rubric: 1/0 for binary, 0/1/2 for ordinal-3, -1 for N/A.

Two backends:
  Gemini mode (default) — GEMINI_API_KEY from keys.env  (pip install google-genai python-dotenv)
  API mode              — ANTHROPIC_API_KEY             (pip install anthropic)

Cache key = MD5(trajectory_id + rubric_id + rubric_description), so editing a
rubric description automatically invalidates its cached scores.
"""

import json
import hashlib
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from keys.env file
load_dotenv(os.environ["KEYS_ENV_PATH"], override=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_BASE = Path(__file__).parent
sys.path.insert(0, str(_BASE.parent / "rewards"))
from rubrics_judge import _build_prompt, _check_precondition as _check_precondition_rj

CACHE_FILE      = Path(__file__).parent / "scores" / "scores_cache.json"
GEMINI_MODEL    = "gemini-2.5-flash-lite"
CLAUDE_MODEL    = "claude-sonnet-4-6"
THINKING_BUDGET = 1024  # set to 0 for models that don't support thinking

MAX_WORKERS  = 4   # concurrent trajectory calls; tune to API rate limits

_cache_lock = threading.Lock()


# ── Cache ──────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _cache_key(trajectory_id: str, rubric_id: str, rubric_description: str) -> str:
    content = f"{trajectory_id}::{rubric_id}::{rubric_description}"
    return hashlib.md5(content.encode()).hexdigest()


# ── Trajectory parsing ─────────────────────────────────────────────────────────

def _extract_trajectory_features(trajectory_path: Path) -> dict:
    turns = []
    with open(trajectory_path) as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))

    think_blocks, search_queries, retrieved_passages, answers = [], [], [], []

    for turn in turns:
        response = turn.get("response", "")
        think_blocks.extend(re.findall(r"<think>(.*?)</think>", response, re.DOTALL))
        search_queries.extend(
            q.strip() for q in re.findall(r"<search>(.*?)</search>", response, re.DOTALL)
        )
        answers.extend(re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL))
        inp = turn.get("input", "")
        retrieved_passages.extend(
            re.findall(r"\[T\d+-R\d+\].*?(?=\[T\d+-R\d+\]|\Z)", inp, re.DOTALL)
        )

    full_text = "\n\n".join(
        [f"[THINK {i+1}]\n{t}" for i, t in enumerate(think_blocks)]
        + [f"[SEARCH {i+1}]\n{q}" for i, q in enumerate(search_queries)]
        + [f"[RETRIEVED {i+1}]\n{p[:400]}" for i, p in enumerate(retrieved_passages)]
        + [f"[ANSWER]\n{a}" for a in answers]
    )

    # Loose citation detection: handles [T1-R2], (T1-R2), T1-R2, [T1-R2, T1-R3], etc.
    # Only scan model-written text (think + answer), not passage headers.
    model_text = "\n\n".join(think_blocks + answers)
    _cited = {
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"T(\d+)-R(\d+)", model_text)
    }

    return {
        "search_performed": len(search_queries) > 0,
        "num_searches":     len(search_queries),
        "has_citations":    len(_cited) > 0,
        "answer":           len(search_queries) > 0 and len(answers) > 0,
        "full_text":        full_text[:6000],
    }


def _check_precondition(rubric: dict, features: dict) -> bool:
    pre = rubric.get("precondition")
    if pre is None:
        return True
    return bool(features.get(pre, False))


# ── LLM call ───────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, grader: str = "gemini",
              max_output_tokens: int = 4096, json_output: bool = True,
              thinking: bool = False) -> str:
    """
    grader="gemini" (default): google.genai client — requires GEMINI_API_KEY in keys.env.
    grader="claude":           anthropic.Anthropic() client — requires ANTHROPIC_API_KEY.
    json_output: set False when the expected output is not JSON (e.g. YAML from drafter).
    thinking: enable Gemini internal thinking budget.
    """
    if grader == "gemini":
        import time
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=THINKING_BUDGET if thinking else 0,
            ),
        )
        if json_output:
            config.response_mime_type = "application/json"
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=config
                )
                return response.text.strip()
            except Exception as e:
                if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                    raise
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s, 160s
                print(f"    [503] {GEMINI_MODEL} overloaded, retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"_call_llm: {GEMINI_MODEL} unavailable after 5 retries")

    # ── Anthropic API backend ─────────────────────────────────────────────
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_output_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()




# ── Per-trajectory batch scorer ────────────────────────────────────────────────

def score_trajectory(
    trajectory_path: Path,
    rubrics: list[dict],
    cache: dict,
    force_rescore: bool = False,
    grader: str = "gemini",
) -> dict[str, int]:
    """
    Score all active rubrics for ONE trajectory in a single LLM call.
    Returns {rubric_id: score}.  Writes results into cache (thread-safe).
    """
    traj_id  = trajectory_path.stem
    features = _extract_trajectory_features(trajectory_path)

    result: dict[str, int] = {}
    to_score: list[dict]   = []

    for rubric in rubrics:
        rid = rubric["id"]
        key = _cache_key(traj_id, rid, rubric["description"])

        # Precondition not met → N/A (-1), no LLM call needed
        if not _check_precondition(rubric, features):
            with _cache_lock:
                cache[key] = {"score": -1, "reasoning": "N/A: precondition not met"}
            result[rid] = -1
            continue

        # Cache hit
        if not force_rescore:
            with _cache_lock:
                cached = cache.get(key)
            if cached is not None:
                result[rid] = cached["score"]
                continue

        to_score.append(rubric)

    if not to_score:
        return result

    # ── Single batched LLM call for all uncached, applicable rubrics ──────
    prompt = _build_prompt(to_score, features, include_reasoning=False)
    try:
        raw    = _call_llm(prompt, grader=grader, thinking=False)
        raw    = re.sub(r"^```json\s*", "", raw)
        raw    = re.sub(r"\s*```$",     "", raw)
        parsed = json.loads(raw)

        for rubric in to_score:
            rid   = rubric["id"]
            key   = _cache_key(traj_id, rid, rubric["description"])
            entry = parsed.get(rid)
            try:
                score = int(entry) if not isinstance(entry, dict) else int(entry["score"])
            except (TypeError, ValueError) as e:
                print(f"    [WARN] {traj_id}/{rid}: missing/invalid in batch response — {e}")
                score = -1
            with _cache_lock:
                cache[key] = {"score": score, "reasoning": ""}
            result[rid] = score

    except Exception as e:
        print(f"    [WARN] {traj_id} batch call failed: {e}")
        for rubric in to_score:
            rid = rubric["id"]
            key = _cache_key(traj_id, rid, rubric["description"])
            with _cache_lock:
                cache[key] = {"score": -1, "reasoning": f"Error: {e}"}
            result[rid] = -1

    return result


# ── Main entry point ───────────────────────────────────────────────────────────

def score_all(
    trajectory_dir: Path,
    rubrics: list[dict],
    force_rescore: bool = False,
    grader: str = "gemini",
    max_trajectories: int | None = None,
    trajectory_ids: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Score all active rubrics on all trajectories.
    One batched LLM call per trajectory, parallelised with ThreadPoolExecutor.
    Returns scores[trajectory_stem][rubric_id] = score (int).

    trajectory_ids: if provided, only score trajectories whose stem is in this set.
    """
    cache            = _load_cache()
    active_rubrics   = [r for r in rubrics if r.get("active", True) and not r.get("needs_redefinition", False)]
    trajectory_files = sorted(trajectory_dir.glob("*.jsonl"))
    if trajectory_ids is not None:
        trajectory_files = [p for p in trajectory_files if p.stem in trajectory_ids]
        missing = trajectory_ids - {p.stem for p in trajectory_files}
        if missing:
            print(f"  [WARN] {len(missing)} trajectory IDs from filter not found in {trajectory_dir}")
    elif max_trajectories is not None and max_trajectories < len(trajectory_files):
        trajectory_files = random.Random(42).sample(trajectory_files, max_trajectories)
        trajectory_files = sorted(trajectory_files)
    backend          = f"Claude ({CLAUDE_MODEL})" if grader == "claude" else f"Gemini ({GEMINI_MODEL})"

    print(f"\nScoring {len(trajectory_files)} trajectories × {len(active_rubrics)} rubrics"
          f"  [{backend}]  [workers={MAX_WORKERS}]...")

    scores: dict[str, dict[str, int]] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_traj = {
            executor.submit(
                score_trajectory,
                traj_path, active_rubrics, cache, force_rescore, grader,
            ): traj_path
            for traj_path in trajectory_files
        }
        for future in as_completed(future_to_traj):
            traj_path = future_to_traj[future]
            traj_id   = traj_path.stem
            try:
                scores[traj_id] = future.result()
            except Exception as e:
                print(f"    [WARN] {traj_id}: {e}")
                scores[traj_id] = {r["id"]: -1 for r in active_rubrics}
            completed += 1
            if completed % 10 == 0 or completed == len(trajectory_files):
                with _cache_lock:
                    _save_cache(cache)
                print(f"    [{completed}/{len(trajectory_files)}] trajectories scored")

    return scores
