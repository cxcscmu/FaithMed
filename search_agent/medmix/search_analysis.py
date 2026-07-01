import argparse
import glob
import json
import os
import re
from collections import defaultdict


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_stub_from_result_name(file_name):
    if not file_name.startswith("result_") or not file_name.endswith(".json"):
        return None
    return file_name[len("result_") : -len(".json")]


def check_format_correct(response_text):
    if not isinstance(response_text, str):
        return False

    start = "<response>"
    end = "</response>"
    if start in response_text and end in response_text:
        # If a response block exists, boxed answer must be inside it.
        answer = response_text.split(start, 1)[1].split(end, 1)[0]
        return re.search(r"\\boxed\{.*?\}", answer, flags=re.DOTALL) is not None

    # If no response block exists, any boxed answer in text is acceptable.
    return re.search(r"\\boxed\{.*?\}", response_text, flags=re.DOTALL) is not None

def load_last_response_from_trajectory(search_log_dir, file_name):
    stub = get_stub_from_result_name(file_name)
    if not stub:
        return ""
    trajectory_path = os.path.join(search_log_dir, f"trajectory_{stub}.jsonl")
    if not os.path.exists(trajectory_path):
        return ""
    last_response = ""
    try:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                response = item.get("response", "")
                if isinstance(response, str):
                    last_response = response
    except Exception:
        return ""
    return last_response


def count_unique_citations(text):
    if not isinstance(text, str):
        return 0
    matches = re.findall(r"\[T(\d+)-R(\d+)\]", text)
    if not matches:
        return 0
    return len({(int(t), int(r)) for t, r in matches})


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "n_score_subset": 0,
            "gen_score": 0.0,
            "base_score": 0.0,
            "delta_score": 0.0,
            "gen_format_correct_count": 0,
            "base_format_correct_count": 0,
            "gen_format_correctness": 0.0,
            "base_format_correctness": 0.0,
            "delta_format_correctness": 0.0,
            "gen_answer_length": 0.0,
            "base_answer_length": 0.0,
            "delta_answer_length": 0.0,
            "gen_search_count": 0.0,
            "gen_reference_count_nonzero_n": 0,
            "gen_reference_proportion": 0.0,
            "gen_reference_count": 0.0,
        }

    gen_format_correct_count = sum(1 for r in rows if r.get("gen_format_correct", False))
    base_format_correct_count = sum(1 for r in rows if r.get("base_format_correct", False))
    score_subset = [r for r in rows if r.get("gen_format_correct", False)]
    n_score_subset = len(score_subset)
    if n_score_subset > 0:
        gen_score = sum(float(r["gen"].get("score", 0.0)) for r in score_subset) / n_score_subset
        base_score = sum(float(r["base"].get("score", 0.0)) for r in score_subset) / n_score_subset
    else:
        gen_score = 0.0
        base_score = 0.0

    gen_answer_length = sum(float(r["gen"].get("answer_length", 0.0)) for r in rows) / n
    base_answer_length = sum(float(r["base"].get("answer_length", 0.0)) for r in rows) / n
    gen_search_count = sum(float(r["gen"].get("search count", 0.0)) for r in rows) / n
    gen_reference_counts = []
    for r in rows:
        gen_record = r.get("gen", {})
        answer_text = gen_record.get("answer", "")
        gen_reference_counts.append(count_unique_citations(answer_text))
    gen_reference_count_nonzero = [c for c in gen_reference_counts if c > 0]
    gen_reference_count_nonzero_n = len(gen_reference_count_nonzero)
    gen_reference_proportion = gen_reference_count_nonzero_n / n
    gen_reference_count = (
        sum(gen_reference_count_nonzero) / gen_reference_count_nonzero_n
        if gen_reference_count_nonzero_n > 0
        else 0.0
    )

    return {
        "n": n,
        "n_score_subset": n_score_subset,
        "gen_score": gen_score,
        "base_score": base_score,
        "delta_score": gen_score - base_score,
        "gen_format_correct_count": gen_format_correct_count,
        "base_format_correct_count": base_format_correct_count,
        "gen_format_correctness": gen_format_correct_count / n,
        "base_format_correctness": base_format_correct_count / n,
        "delta_format_correctness": (gen_format_correct_count - base_format_correct_count) / n,
        "gen_answer_length": gen_answer_length,
        "base_answer_length": base_answer_length,
        "delta_answer_length": gen_answer_length - base_answer_length,
        "gen_search_count": gen_search_count,
        "gen_reference_count_nonzero_n": gen_reference_count_nonzero_n,
        "gen_reference_proportion": gen_reference_proportion,
        "gen_reference_count": gen_reference_count,
    }


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def print_block(title, s):
    print(title)
    print(f"  n: {s['n']}")
    print(
        f"  score (agent format-correct subset, n={s['n_score_subset']}): search={s['gen_score']:.4f}, base={s['base_score']:.4f}, delta={s['delta_score']:+.4f}"
    )
    print(
        f"  format_correctness: search={fmt_pct(s['gen_format_correctness'])} ({s['gen_format_correct_count']}/{s['n']}), base={fmt_pct(s['base_format_correctness'])} ({s['base_format_correct_count']}/{s['n']}), delta={fmt_pct(s['delta_format_correctness'])}"
    )
    print(
        f"  answer_length: search={s['gen_answer_length']:.2f}, base={s['base_answer_length']:.2f}, delta={s['delta_answer_length']:+.2f}"
    )
    print(f"  avg_search_count(search-run only): {s['gen_search_count']:.2f}")
    print(
        f"  reference_proportion(search-run only): {fmt_pct(s['gen_reference_proportion'])} ({s['gen_reference_count_nonzero_n']}/{s['n']})"
    )
    print(
        f"  reference_count(search-run only, among referenced answers): {s['gen_reference_count']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare generate vs generate_base on samples that used search tool."
    )
    parser.add_argument("--search_answer_dir", required=True, help="answer_dir from generate.py")
    parser.add_argument("--base_answer_dir", required=True, help="answer_dir from generate_base.py")
    parser.add_argument("--search_log_dir", required=True, help="log_dir from generate.py (contains trajectory_*.jsonl)")
    args = parser.parse_args()

    search_files = sorted(glob.glob(os.path.join(args.search_answer_dir, "result_*.json")))
    if not search_files:
        print(f"No result files found in search dir: {args.search_answer_dir}")
        return

    matched_rows = []
    missing_in_base = []
    missing_trajectory = 0
    skipped_no_search = 0

    for search_path in search_files:
        gen = load_json(search_path)
        sc = float(gen.get("search count", 0.0))
        if sc == 0.0:
            skipped_no_search += 1
            continue

        file_name = os.path.basename(search_path)
        base_path = os.path.join(args.base_answer_dir, file_name)
        base = load_json(base_path)
        if base is None:
            missing_in_base.append(file_name)
            continue

        final_response = load_last_response_from_trajectory(args.search_log_dir, file_name)
        gen_format_correct = check_format_correct(final_response)
        if not final_response:
            missing_trajectory += 1
        base_format_correct = base.get("complete", False)

        matched_rows.append(
            {
                "file_name": file_name,
                "gen": gen,
                "base": base,
                "gen_format_correct": gen_format_correct,
                "base_format_correct": base_format_correct,
            }
        )

    print("=== Search-vs-Base Analysis (search-triggered subset) ===")
    print(f"search_answer_dir: {args.search_answer_dir}")
    print(f"base_answer_dir:   {args.base_answer_dir}")
    print(f"search_log_dir:    {args.search_log_dir}")
    print(f"total_search_run_files: {len(search_files)}")
    print(f"skipped_not_enough_search: {skipped_no_search}")
    print(f"matched_pairs: {len(matched_rows)}")
    print(f"missing_in_base: {len(missing_in_base)}")
    print(f"missing_or_unreadable_trajectory: {missing_trajectory}")
    if missing_in_base:
        preview = ", ".join(missing_in_base[:10])
        print(f"missing_examples_preview(<=10): {preview}")
    print("")

    if not matched_rows:
        print("No matched pairs to compare.")
        return

    by_source = defaultdict(list)
    for row in matched_rows:
        source = str(row["gen"].get("data_source", "unknown"))
        by_source[source].append(row)

    overall = summarize(matched_rows)
    print_block("[Overall]", overall)
    print("")

    for source in sorted(by_source.keys()):
        s = summarize(by_source[source])
        print_block(f"[{source}]", s)
        print("")


if __name__ == "__main__":
    main()
