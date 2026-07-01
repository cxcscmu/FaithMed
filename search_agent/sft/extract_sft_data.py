"""
Step 2 of 2: Extract LLaMA-Factory SFT data from saved trajectories.

Reads the per-question JSONL files produced by generate_trajectories.py (no API
calls required) and writes a single LLaMA-Factory alpaca JSONL ready for training.

Formatting fixes applied at this stage: Citation normalisation: bare ``T2-R3`` and ``(T2-R3)`` → ``[T2-R3]``

Output format (LLaMA-Factory alpaca):
  {"instruction": "<full accumulated prompt>", "input": "", "output": "<think>...</think>\\n<action>"}

  instruction  – the full prompt at that turn (history + question).  Set
                 ``train_on_prompt: false`` in LLaMA-Factory so only the output
                 contributes to the cross-entropy loss.
  output       – the teacher's complete response including the <think> block, with
                 citation formatting corrected.

Usage:
  python -m search_agent.sft.extract_sft_data \\
      --answer_dir   /path/to/trajectories \\
      --output_path  /path/to/sft_data.jsonl \\
      --max_per_question 6

LLaMA-Factory dataset_info.json entry:
  "medmix_sft": {
    "file_name": "sft_data.jsonl",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "response": "output"}
  }
Then in the training YAML: train_on_prompt: false
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Formatting fixes
# ---------------------------------------------------------------------------

_BARE_CITATION_RE = re.compile(r"(?<![(\[])T(\d+)-R(\d+)(?![)\]])")


def fix_minor_formatting(raw_response: str) -> str:
    """
    Fix minor formatting issues in the teacher's raw response before saving as
    the SFT training target.

    Current fixes:
      1. ``(T2-R3)`` → ``[T2-R3]``   (parenthesis-wrapped citations → brackets)
      2. bare ``T2-R3`` → ``[T2-R3]`` (unwrapped citations → brackets)
    """
    if not raw_response:
        return raw_response
    # Paren-wrapped first so the bare-citation pattern does not see the parens
    text = re.sub(r"\(T(\d+)-R(\d+)\)", r"[T\1-R\2]", raw_response)
    text = _BARE_CITATION_RE.sub(r"[T\1-R\2]", text)
    return text


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_trajectories(log_dir: str) -> list[dict]:
    """
    Load every trajectory from ``log_dir/accepted/trajectory_*.jsonl``.

    Each trajectory is a list of turn dicts.  The parent (LLMAgent) writes one
    JSONL file per trajectory attempt (one line per turn) when verbose=True:
      {"input": "<prompt>", "response": "<raw_response>", "next_obs": "...", "context_length": N}

    Returns a list of trajectory dicts:
      {"file": str, "turns": [{"input": ..., "response": ...}, ...]}
    """
    accepted_dir = os.path.join(log_dir, "accepted")
    pattern = os.path.join(accepted_dir, "trajectory_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No accepted trajectory files found in {accepted_dir}. "
            "Run generate_trajectories.py first."
        )

    trajectories = []
    for fpath in files:
        turns = []
        with open(fpath, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: skipping malformed line {lineno} in {fpath}: {e}")
        if turns:
            trajectories.append({"file": fpath, "turns": turns})
    return trajectories


# ---------------------------------------------------------------------------
# Filtering / rejection sampling
# ---------------------------------------------------------------------------

def cap_trajectories_per_question(
    trajectories: list[dict],
    max_per_question: int = 6,
) -> list[dict]:
    """
    Cap at max_per_question trajectories per question (by filename prefix).

    All trajectories in log_dir/accepted/ already passed rejection sampling at
    generation time (score == 1.0, no invalid turns).  This function only applies
    the per-question cap, taking the first N files in sorted order.
    """
    by_question: dict[str, list[dict]] = defaultdict(list)
    for traj in trajectories:
        # Question key: stub without the attempt suffix (_a##.jsonl → stub)
        fname = os.path.basename(traj["file"])           # trajectory_{stub}_a03.jsonl
        stub  = re.sub(r"_a\d+\.jsonl$", "", fname[len("trajectory_"):])
        by_question[stub].append(traj)

    capped = []
    for stub, trajs in by_question.items():
        trajs_sorted = sorted(trajs, key=lambda t: t["file"])
        capped.extend(trajs_sorted[:max_per_question])
    return capped


# ---------------------------------------------------------------------------
# Conversion to SFT samples
# ---------------------------------------------------------------------------

def trajectory_to_sft_samples(trajectory: dict) -> list[dict]:
    """
    Convert one accepted trajectory to a list of LLaMA-Factory alpaca samples.

    The parent (LLMAgent) writes per-turn JSONL with fields:
      "input"    → full accumulated prompt  (SFT instruction, no loss)
      "response" → complete model output including <think>...</think>  (SFT output)

    Every turn in an accepted trajectory is trainable: generate_trajectories.py
    rejects trajectories that had any format-error turn, so no per-turn filtering
    is needed here.

    Citation formatting is normalised before saving.
    """
    samples = []
    for turn in trajectory.get("turns", []):
        output = fix_minor_formatting(turn.get("response") or "")
        if not output:
            continue
        samples.append({
            "instruction": turn["input"],
            "input": "",
            "output": output,
        })
    return samples


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_sft_data(
    log_dir: str,
    output_path: str,
    max_per_question: int = 6,
):
    """
    Full pipeline: load accepted trajectories → cap per question → fix formatting
    → write LLaMA-Factory alpaca JSONL.

    Reads from log_dir/accepted/trajectory_*.jsonl (written by generate_trajectories.py
    with verbose=True).  All files in that folder already passed rejection sampling
    at generation time (score==1.0, no invalid turns).
    """
    print(f"Loading accepted trajectories from: {os.path.join(log_dir, 'accepted')}")
    all_trajs = load_all_trajectories(log_dir)
    print(f"  Found {len(all_trajs)} accepted trajectory files.")

    capped = cap_trajectories_per_question(all_trajs, max_per_question=max_per_question)
    n_questions = len({
        re.sub(r"_a\d+\.jsonl$", "", os.path.basename(t["file"])[len("trajectory_"):])
        for t in capped
    })
    print(f"  After cap (max_per_q={max_per_question}): "
          f"{len(capped)} trajectories across {n_questions} questions.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total_samples = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for traj in capped:
            samples = trajectory_to_sft_samples(traj)
            for s in samples:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
            total_samples += len(samples)

    print(f"\n===== Extraction complete =====")
    print(f"  SFT samples written : {total_samples}")
    print(f"  Output              : {output_path}")

    stats = {
        "accepted_trajectories": len(all_trajs),
        "after_cap": len(capped),
        "questions_covered": n_questions,
        "sft_samples": total_samples,
        "max_per_question": max_per_question,
    }
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats               : {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract LLaMA-Factory SFT data from trajectory files.")
    p.add_argument("--log_dir", required=True,
                   help="log_dir used in generate_trajectories.py (contains accepted/ subfolder)")
    p.add_argument("--output_path", required=True,
                   help="Output path for LLaMA-Factory alpaca JSONL")
    p.add_argument("--max_per_question", type=int, default=6,
                   help="Maximum trajectories per question (default 6)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_sft_data(
        log_dir=args.log_dir,
        output_path=args.output_path,
        max_per_question=args.max_per_question,
    )
