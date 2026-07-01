"""
Step 1 of 2: Generate and save multi-trajectory data using a teacher model.

After each attempt, files are routed to one of three folders:

  accepted/      — correct (score == 1.0, no format errors) AND the question has
                   accumulated num_trajectories such attempts.

  hard_question/ — individually correct trajectories (same per-attempt criterion as
                   accepted/) but the question exhausted max_attempts without ever
                   reaching num_trajectories successes.  

  rejected/      — wrong answer (score < 1.0) OR at least one invalid-format turn,
                   regardless of question difficulty.

Output layout:
  save_dir/
    logs/
      accepted/
        trajectory_{data_source}_{sample_id}_a{N}.jsonl   # per-turn JSONL
        trajectory_{data_source}_{sample_id}_a{N}.md      # human-readable
      hard_question/
        trajectory_{data_source}_{sample_id}_a{N}.jsonl
        trajectory_{data_source}_{sample_id}_a{N}.md
      rejected/
        trajectory_{data_source}_{sample_id}_a{N}.jsonl
        trajectory_{data_source}_{sample_id}_a{N}.md
    answers/
      accepted/
        result_{data_source}_{sample_id}_a{N}.json        # result + score
      hard_question/
        result_{data_source}_{sample_id}_a{N}.json
      rejected/
        result_{data_source}_{sample_id}_a{N}.json

Usage:
  python -m search_agent.sft.generate_trajectories \\
      --data_path  /path/to/medmix/train.parquet \\
      --save_dir   /path/to/trajectories \\
      --model bedrock \\
      --search_engine medcorp \\
      --temperature 0.8 \\
      --num_trajectories 6 \\
      --max_attempts 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import traceback
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from search_agent.medmix.generate_agent_search import (
    LLMAgent,
    load_samples_from_medmix_parquet,
)

load_dotenv(os.environ.get("KEYS_ENV_PATH", "keys.env"), override=False)

CONCURRENT_NUM = 16


class TrajectoryCollector(LLMAgent):
    """
    Thin subclass of LLMAgent for SFT trajectory collection.

    All trajectory execution and saving reuses the parent's run_llm_loop unchanged.
    New logic added here:
      1. temperature > 0 — inherited from LLMAgent now that it stores self.temperature.
      2. Invalid-turn tracking — _execute_response override sets a flag when the
         model produces a format error on any turn.
      3. File routing — after each run_llm_loop call, result + trajectory files are
         moved to accepted/ or rejected/ based on score and the invalid-turn flag.
      4. Multi-attempt loop — run_question runs up to max_attempts per question with
         resume support.
    """

    def __init__(
        self,
        config: dict,
        num_trajectories: int = 6,
        max_attempts: int = 20,
        **parent_kwargs,
    ):
        # temperature is passed via parent_kwargs and stored as self.temperature
        # by the parent __init__ (added to LLMAgent)
        super().__init__(config=config, **parent_kwargs)
        self.num_trajectories = num_trajectories
        self.max_attempts = max_attempts
        self.had_invalid_turn: dict = {}  # question_id -> bool, reset each attempt

    # ------------------------------------------------------------------
    # Track invalid turns (format errors) without changing parent logic
    # ------------------------------------------------------------------

    def _execute_response(self, processed_response, num_docs, question_id,
                          search_log, do_search=True, raw_response=None):
        if processed_response is None:
            self.had_invalid_turn[question_id] = True
        return super()._execute_response(
            processed_response, num_docs, question_id, search_log, do_search, raw_response
        )

    # ------------------------------------------------------------------
    # File routing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attempt_stub(file_stub: str, attempt_idx: int) -> str:
        return f"{file_stub}_a{attempt_idx:02d}"

    def _count_accepted(self, file_stub: str) -> int:
        # Count accepted files in both accepted/ and hard_question/ so resume
        # correctly picks up the prior accepted count for hard questions.
        n = len(glob.glob(os.path.join(self.answer_dir, "accepted",      f"result_{file_stub}_a*.json")))
        n += len(glob.glob(os.path.join(self.answer_dir, "hard_question", f"result_{file_stub}_a*.json")))
        return n

    def _count_total_attempts(self, file_stub: str) -> int:
        acc  = len(glob.glob(os.path.join(self.answer_dir, "accepted",      f"result_{file_stub}_a*.json")))
        rej  = len(glob.glob(os.path.join(self.answer_dir, "rejected",      f"result_{file_stub}_a*.json")))
        hard = len(glob.glob(os.path.join(self.answer_dir, "hard_question", f"result_{file_stub}_a*.json")))
        return acc + rej + hard

    def _route_files(self, attempt_stub: str, is_accepted: bool):
        """Move result JSON and trajectory files into accepted/ or rejected/ subfolder."""
        dest = "accepted" if is_accepted else "rejected"
        for base_dir, fname in [
            (self.log_dir,    f"trajectory_{attempt_stub}.md"),
            (self.log_dir,    f"trajectory_{attempt_stub}.jsonl"),
            (self.log_dir,    f"search_{attempt_stub}.log"),
            (self.answer_dir, f"result_{attempt_stub}.json"),
        ]:
            src = os.path.join(base_dir, fname)
            if os.path.exists(src):
                dest_dir = os.path.join(base_dir, dest)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(src, os.path.join(dest_dir, fname))

    # ------------------------------------------------------------------
    # Per-question multi-attempt loop
    # ------------------------------------------------------------------

    def run_question(self, sample: dict) -> tuple[int, int]:
        """
        Run up to max_attempts trajectories for one question.
        Resumes from existing accepted trajectories if the job was interrupted.

        Returns:
            (num_newly_accepted, num_rejected_this_run)
        """
        question_id = sample["question_id"]
        file_stub   = sample["file_stub"]

        already_accepted = self._count_accepted(file_stub)
        if already_accepted >= self.num_trajectories:
            print(f"  [q={question_id}] already has {already_accepted} accepted – skipped.")
            return 0, 0, True

        accepted = already_accepted
        rejected = 0
        attempt_idx = self._count_total_attempts(file_stub)

        for _ in range(attempt_idx, self.max_attempts):
            if accepted >= self.num_trajectories:
                break

            a_stub = self._attempt_stub(file_stub, attempt_idx)

            # Swap in the attempt-indexed file_stub so run_llm_loop saves to
            # distinct files for each attempt (e.g. result_medqa_q1_a03.json).
            attempt_sample = {**sample, "file_stub": a_stub}
            self.questions[question_id]   = attempt_sample["question"]
            self.prompt_ids[question_id]  = attempt_sample.get("prompt_id", question_id)
            self.sample_meta[question_id] = attempt_sample
            self.had_invalid_turn[question_id] = False  # reset for this attempt

            # Run the full trajectory — parent handles execution, logging, and saving
            self.run_llm_loop(sample["task_description"], question_id)

            # Read score from the result file the parent just wrote
            result_path = os.path.join(self.answer_dir, f"result_{a_stub}.json")
            try:
                with open(result_path, encoding="utf-8") as f:
                    score = json.load(f).get("score", 0.0)
            except Exception:
                score = 0.0

            is_accepted = score == 1.0 and not self.had_invalid_turn.get(question_id, False)
            self._route_files(a_stub, is_accepted)
            attempt_idx += 1

            if is_accepted:
                accepted += 1
                print(
                    f"  [q={question_id}] attempt {attempt_idx}: ACCEPTED "
                    f"({accepted}/{self.num_trajectories})"
                )
            else:
                rejected += 1
                reason = "invalid_turns" if score == 1.0 else "wrong_answer"
                print(
                    f"  [q={question_id}] attempt {attempt_idx}: rejected "
                    f"({reason}, score={score:.2f})"
                )

        fully_satisfied = accepted >= self.num_trajectories
        if not fully_satisfied:
            print(
                f"  [q={question_id}] HARD QUESTION – only {accepted}/"
                f"{self.num_trajectories} accepted after {attempt_idx} attempts."
            )
            # Move any accepted files for this question out of accepted/ into
            # hard_question/ so that accepted/ only contains fully-satisfied
            # questions.  hard_question/ is included in both _count_accepted and
            # _count_total_attempts so resume correctly picks up from here.
            safe_stub = sample["file_stub"].replace("/", "_")
            for base_dir in [self.log_dir, self.answer_dir]:
                src_dir  = os.path.join(base_dir, "accepted")
                dest_dir = os.path.join(base_dir, "hard_question")
                if not os.path.isdir(src_dir):
                    continue
                for fname in os.listdir(src_dir):
                    if fname.endswith((".jsonl", ".md", ".json", ".log")) and safe_stub in fname:
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.move(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

        return accepted - already_accepted, rejected, fully_satisfied

    # ------------------------------------------------------------------
    # Parallel execution across questions
    # ------------------------------------------------------------------

    def run_parallel(self, samples: list[dict]):
        print(f"Running {len(samples)} questions, up to {CONCURRENT_NUM} in parallel.")
        stats = {"total": len(samples), "accepted": 0, "rejected": 0, "hard": 0}

        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = {executor.submit(self.run_question, s): s for s in samples}
            with tqdm(total=len(samples), desc="Questions", unit="q") as pbar:
                for future in as_completed(futures):
                    try:
                        acc, rej, fully_satisfied = future.result()
                        stats["accepted"] += acc
                        stats["rejected"] += rej
                        if not fully_satisfied:
                            stats["hard"] += 1
                    except Exception:
                        print(traceback.format_exc())
                        stats["hard"] += 1
                    pbar.update(1)

        print("\n===== Trajectory generation complete =====")
        print(f"  Questions total    : {stats['total']}")
        print(f"  New accepted trajs : {stats['accepted']}")
        print(f"  Rejected trajs     : {stats['rejected']}")
        print(f"  Hard questions     : {stats['hard']}")
        print(f"  Answer dir         : {self.answer_dir}")
        print(f"  Log dir            : {self.log_dir}")

def filter_completed_samples(
    samples: list[dict], answer_dir: str, num_trajectories: int, max_attempts: int
) -> tuple[list, int]:
    """Return samples that still need work.

    A question is considered done if either:
      - it has >= num_trajectories accepted trajectories, OR
      - it has been attempted >= max_attempts times in total (hard question, no more retries).
    """
    pending, done = [], 0
    for s in samples:
        stub = s["file_stub"]
        n_accepted = len(glob.glob(os.path.join(answer_dir, "accepted",      f"result_{stub}_a*.json")))
        n_total    = n_accepted
        n_total   += len(glob.glob(os.path.join(answer_dir, "rejected",      f"result_{stub}_a*.json")))
        n_total   += len(glob.glob(os.path.join(answer_dir, "hard_question", f"result_{stub}_a*.json")))
        if n_accepted >= num_trajectories or n_total >= max_attempts:
            done += 1
        else:
            pending.append(s)
    return pending, done

def save_final_stats(answer_dir: str, save_dir: str, num_trajectories: int, max_attempts: int) -> None:
    """Scan all three answer subfolders and write a cumulative final_stats.jsonl.

    Output format (one JSON object per line):
      {"type": "overall_summary", ...}          # aggregate across all sources
      {"type": "source_summary", "data_source": ..., ...}  # one per data source
      {"type": "question", "data_source": ..., "question": ..., ...}  # one per question
    """
    from collections import defaultdict

    # Map sanitized filename prefix -> original data_source name
    source_prefixes = [
        (s.replace("/", "_"), s) for s in _ORDERED_SOURCES
    ]

    def stub_to_source(stub: str) -> str:
        for prefix, source in source_prefixes:
            if stub.startswith(prefix):
                return source
        return "unknown"

    # Count attempts per stub across all folders
    counts: dict = defaultdict(lambda: {"accepted": 0, "hard_question": 0, "rejected": 0})
    for folder in ("accepted", "hard_question", "rejected"):
        for fpath in glob.glob(os.path.join(answer_dir, folder, "result_*_a*.json")):
            stub = re.sub(r"_a\d+\.json$", "", os.path.basename(fpath)[len("result_"):])
            counts[stub][folder] += 1

    # Build per-question rows grouped by data_source
    by_source: dict = defaultdict(list)
    for stub, c in sorted(counts.items()):
        n_acc   = c["accepted"]
        n_hq    = c["hard_question"]
        n_rej   = c["rejected"]
        n_total = n_acc + n_hq + n_rej
        if n_acc >= num_trajectories:
            status = "accepted"
        elif n_total >= max_attempts:
            status = "hard_question"
        else:
            status = "pending"
        by_source[stub_to_source(stub)].append({
            "type":                 "question",
            "data_source":          stub_to_source(stub),
            "question":             stub,
            "status":               status,
            "accepted_trajs":       n_acc,
            "hard_question_trajs":  n_hq,
            "rejected_trajs":       n_rej,
            "total_attempts":       n_total,
        })

    def _agg(rows: list[dict]) -> dict:
        return {
            "questions_total":           len(rows),
            "fully_accepted_questions":  sum(r["status"] == "accepted"       for r in rows),
            "hard_questions":            sum(r["status"] == "hard_question"   for r in rows),
            "pending_questions":         sum(r["status"] == "pending"         for r in rows),
            "total_accepted_trajs":      sum(r["accepted_trajs"]              for r in rows),
            "total_hard_question_trajs": sum(r["hard_question_trajs"]         for r in rows),
            "total_rejected_trajs":      sum(r["rejected_trajs"]              for r in rows),
        }

    all_rows = [r for rows in by_source.values() for r in rows]
    overall  = {"type": "overall_summary", **_agg(all_rows)}

    out_path = os.path.join(save_dir, "final_stats.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(overall) + "\n")
        for source in _ORDERED_SOURCES:
            rows = by_source.get(source, [])
            if rows:
                f.write(json.dumps({"type": "source_summary", "data_source": source, **_agg(rows)}) + "\n")

    print(f"\n  [STATS] Cumulative stats saved → {out_path}")
    print(f"          Overall   : {overall['questions_total']} questions  "
          f"(accepted={overall['fully_accepted_questions']}, "
          f"hard={overall['hard_questions']}, pending={overall['pending_questions']})")
    for source in _ORDERED_SOURCES:
        rows = by_source.get(source, [])
        if rows:
            s = _agg(rows)
            print(f"          {source:<40s}: "
                  f"accepted={s['fully_accepted_questions']}, "
                  f"hard={s['hard_questions']}, pending={s['pending_questions']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories using a teacher model.")
    parser.add_argument("--data_path", default="/path/to/data/medmix/train.parquet")
    parser.add_argument("--save_dir", default="/path/to/SFT")
    parser.add_argument("--model", default="bedrock", choices=["bedrock", "qwen", "gemini", "gpt"])
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--search_engine", default="medcorp", choices=["medcorp", "clueweb", "serper", "fineweb"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_trajectories", type=int, default=6)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--num_docs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sample_limit", type=int, nargs=4, metavar=("MEDQA", "HEAD_QA", "MEDMCQA", "MEDCALC"), default=None, help="Per-source sample limits in fixed order")
    parser.add_argument("--data_source", nargs="+", default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    return parser.parse_args()


_ORDERED_SOURCES = [
    "openlifescienceai/medqa",
    "dvilares/head_qa",
    "openlifescienceai/medmcqa",
    "ncbi/MedCalc-Bench",
]


if __name__ == "__main__":
    args = parse_args()

    answer_dir = os.path.join(args.save_dir, "answers")
    log_dir    = os.path.join(args.save_dir, "logs")
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    config = {"max_turns": args.max_turns, "num_docs": args.num_docs}

    import search_agent.sft.generate_trajectories as _self
    _self.CONCURRENT_NUM = args.workers

    collector = TrajectoryCollector(
        config=config,
        log_dir=log_dir,
        answer_dir=answer_dir,
        model=args.model,
        url=args.url,
        search_engine=args.search_engine,
        temperature=args.temperature,
        num_trajectories=args.num_trajectories,
        max_attempts=args.max_attempts,
        verbose=True,   # must be True to write per-turn trajectory JSONL
    )

    sample_limit = (
        dict(zip(_ORDERED_SOURCES, args.sample_limit))
        if args.sample_limit is not None else None
    )
    samples = load_samples_from_medmix_parquet(
        args.data_path,
        sample_limit=sample_limit,
        sample_seed=args.sample_seed,
        data_sources=args.data_source,
    )

    samples, already_done = filter_completed_samples(samples, answer_dir, args.num_trajectories, args.max_attempts)
    print(f"Loaded {len(samples) + already_done} questions; "
          f"{already_done} already complete, {len(samples)} pending.")

    collector.run_parallel(samples)
    save_final_stats(answer_dir, args.save_dir, args.num_trajectories, args.max_attempts)
