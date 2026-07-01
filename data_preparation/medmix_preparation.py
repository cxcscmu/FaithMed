import os
import argparse
import json
from collections import defaultdict

import datasets

QA_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions.
QUESTION: {task_description}

Let's think step by step and include the final choice in the box \\boxed{{}}."""

CALC_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions.
QUESTION: {task_description}

When you are finished with all computations, please write your final answer value without any units, using the following formats in the boxed form \\boxed{{}}:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)"""

def build_question_with_options(question, options) -> str:
    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key} {options[key]}")
    options_block = "; ".join(lines)
    return f"{question}\n{options_block}"


def build_question_with_note(question, note) -> str:
    return f"Patient Note:\n{note}\n\nQuestion:\n{question}"


def make_medqa_map_fn(data_source, split):
    def process_fn(example, idx):
        raw = example["data"]
        question = raw.get("Question")
        correct_option = raw.get("Correct Option")
        options = raw.get("Options")
        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(example["id"]),
                "question": question_with_options,
            },
        }

    return process_fn


def make_headqa_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("qtext")
        answers = example.get("answers") # list of answer options
        ra = example.get("ra") # right answer id, e.g., "1", "2", ...
        category = example.get("category")
        qid = str(example.get("qid"))
        year = example.get("year")
        name = example.get("name")

        answers_sorted = sorted(answers, key=lambda a: int(a.get("aid", 0)))
        options = {}
        correct_letter = None
        # change answer format to options with letters
        for i, ans in enumerate(answers_sorted):
            letter = chr(ord("A") + i)
            options[letter] = ans.get("atext")
            if ra is not None and str(ans.get("aid")) == str(ra):
                correct_letter = letter

        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_letter},
            "extra_info": {
                "split": split,
                "qid": name + '_' + qid,
                "category": category,
                "year": year,
                "name": name,
                "question": question_with_options,
            },
        }

    return process_fn


def make_medmcqa_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        options = {
            "A": example.get("opa"),
            "B": example.get("opb"),
            "C": example.get("opc"),
            "D": example.get("opd"),
        }
        cop = example.get("cop")  # correct option
        idx_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_option = idx_map[int(cop)]

        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(example.get("id")),
                "subject_name": example.get("subject_name"),
                "topic_name": example.get("topic_name"),
                "choice_type": example.get("choice_type"),
                "explanation": example.get("exp"),
                "question": question_with_options,
            },
        }

    return process_fn


def make_medcalcbench_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("Question")
        note = example.get("Patient Note")
        answer = example.get("Ground Truth Answer")
        explanation = example.get("Ground Truth Explanation")

        question_with_note = build_question_with_note(question, note)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": calc_system_content},
                {"role": "user", "content": CALC_BASE_TEMPLATE.format(task_description=question_with_note)},
                # {"role": "user", "content": question_with_note},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "qid": str(example.get("Note ID")),
                "lower_bound": example.get("Lower Limit"),
                "upper_bound": example.get("Upper Limit"),
                "calculator_id": example.get("Calculator ID"),
                "calculator_name": example.get("Calculator Name"),
                "category": example.get("Category"),
                "output_type": example.get("Output Type"),
                "ground_truth_explanation": explanation,
                "row_number": example.get("Row Number"),
                "question": question_with_note,
            },
        }

    return process_fn


def make_medbullets_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        options = {
            "A": example.get("choicesA"),
            "B": example.get("choicesB"),
            "C": example.get("choicesC"),
            "D": example.get("choicesD"),
            "E": example.get("choicesE"),
        }
        answer_idx = example.get("answer_idx")
        answer_text = example.get("answer")

        correct_option = str(answer_idx).strip().upper()
        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(idx),
                "answer_text": answer_text,
                "explanation": example.get("explanation"),
                "question": question_with_options,
            },
        }

    return process_fn


def make_mmlu_pro_health_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        options_raw = example.get("options")
        answer = example.get("answer")
        answer_index = example.get("answer_index")
        category = example.get("category") or example.get("subject")
        question_id = example.get("question_id") or example.get("id")

        options = {chr(ord("A") + i): text for i, text in enumerate(options_raw)}
        correct_option = str(answer).strip().upper()
        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(idx),
                "answer_index": answer_index,
                "question_id": question_id,
                "category": category,
                "question": question_with_options,
            },
        }

    return process_fn


def make_medxpertqa_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        # options = example.get("options")
        correct_option = example.get("label")

        # question_with_options = f"{question}\n{user_content_suffix}"

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(idx),
                "id": example.get("id"),
                "medical_task": example.get("medical_task"),
                "body_system": example.get("body_system"),
                "question_type": example.get("question_type"),
                "question": question,
            },
        }

    return process_fn


def collect_train_splits(dataset):
    splits = []
    for split_name in ["train", "validation", "valid", "dev"]:
        if split_name in dataset:
            splits.append((split_name, dataset[split_name]))
    return splits


def collect_test_split(dataset):
    if "test" in dataset:
        return "test", dataset["test"]
    raise ValueError("Dataset does not contain a test split.")


def load_dataset_splits(data_src, config=None):
    if config:
        dataset = datasets.load_dataset(data_src, config)
    else:
        dataset = datasets.load_dataset(data_src)
    return collect_train_splits(dataset), collect_test_split(dataset)


def filter_headqa_no_image(split):
    split = split.cast_column("image", datasets.Image(decode=False))
    return split.filter(lambda x: x["image"] is None)


def normalize_question_text(text):
    if not text:
        return ""
    return " ".join(str(text).lower().split())


def collect_ngrams_from_questions(dataset, n=16):
    ngrams = set()
    for example in dataset:
        question = example.get("extra_info", {}).get("question")
        question = normalize_question_text(question)
        if not question:
            continue
        tokens = question.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))
    return ngrams


def has_ngram_overlap(question, ngrams, n=16):
    question = normalize_question_text(question)
    if not question:
        return False
    tokens = question.split()
    if len(tokens) < n:
        return False
    for i in range(len(tokens) - n + 1):
        if " ".join(tokens[i : i + n]) in ngrams:
            return True
    return False


def cap_dataset_per_source(dataset, per_source_limit):
    kept_indices = []
    source_counts = defaultdict(int)
    for idx, example in enumerate(dataset):
        source = example.get("data_source")
        if source_counts[source] < per_source_limit:
            kept_indices.append(idx)
            source_counts[source] += 1
    return dataset.select(kept_indices), dict(source_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, help="data path to save the parquet files")
    parser.add_argument("--output_name", default="medmix", help="output folder name under save_dir")
    parser.add_argument("--train_limit", action="store_true", help="limit final train rows to 10000 per data_source")
    parser.add_argument("--test_limit", action="store_true", help="limit final test rows to 1000 per data_source")
    
    args = parser.parse_args()
    output_dir = os.path.join(args.save_dir, args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "preprocess.log")
    log_lines = []

    def select_splits(dataset, split_names):
        return [(name, dataset[name]) for name in split_names if name in dataset]

    def filter_mmlu_health(split):
        return split.filter(lambda x: (x.get("category") or "").lower() == "health")

    dataset_specs = [
        {
            "name": "openlifescienceai/medqa",
            "config": None,
            "map_fn": make_medqa_map_fn,
            "train_splits": ["train", "dev"],
            "test_splits": ["test"],
        },
        {
            "name": "dvilares/head_qa",
            "config": "en",
            "map_fn": make_headqa_map_fn,
            "train_splits": ["train", "validation"],
            "test_splits": ["test"],
            "filter_fn": filter_headqa_no_image,
            "filter_splits": ["train", "validation", "test"],
        },
        {
            "name": "openlifescienceai/medmcqa",
            "config": None,
            "map_fn": make_medmcqa_map_fn,
            "train_splits": ["train"],
            "test_splits": ["validation"],
        },
        {
            "name": "ncbi/MedCalc-Bench",
            "config": None,
            "map_fn": make_medcalcbench_map_fn,
            "train_splits": ["train"],
            "test_splits": ["test"],
        },
        {
            "name": "JesseLiu/medbulltes5op",
            "config": None,
            "map_fn": make_medbullets_map_fn,
            "train_splits": [],
            "test_splits": ["test"],
        },
        {
            "name": "TIGER-Lab/MMLU-Pro",
            "config": None,
            "map_fn": make_mmlu_pro_health_map_fn,
            "train_splits": [],
            "test_splits": ["test"],
            "filter_fn": filter_mmlu_health,
            "filter_splits": ["test"],
        },
        {
            "name": "TsinghuaC3I/MedXpertQA",
            "config": "Text",
            "map_fn": make_medxpertqa_map_fn,
            "train_splits": [],
            "test_splits": ["test"],
        },
    ]

    train_parts = []
    test_parts = []

    for spec in dataset_specs:
        print(f"processing dataset: {spec['name']}")
        dataset = datasets.load_dataset(spec["name"], spec["config"])
        last_example = None
        for split_name, train_split in select_splits(dataset, spec["train_splits"]):
            if spec.get("filter_fn") and split_name in spec.get("filter_splits", []):
                train_split = spec["filter_fn"](train_split)
            log_lines.append(
                f"adding {len(train_split)} rows from {spec['name']} dataset {split_name} split"
            )
            mapped = train_split.map(
                function=spec["map_fn"](spec["name"], "train"),
                with_indices=True,
                remove_columns=train_split.column_names,
            )
            train_parts.append(mapped)
            if len(mapped) > 0:
                last_example = mapped[-1]
        for split_name, test_split in select_splits(dataset, spec["test_splits"]):
            if spec.get("filter_fn") and split_name in spec.get("filter_splits", []):
                test_split = spec["filter_fn"](test_split)
            log_lines.append(
                f"adding {len(test_split)} rows from {spec['name']} dataset {split_name} split"
            )
            mapped_test = test_split.map(
                function=spec["map_fn"](spec["name"], split_name),
                with_indices=True,
                remove_columns=test_split.column_names,
            )
            test_parts.append(mapped_test)
            if len(mapped_test) > 0:
                last_example = mapped_test[-1]
        # log last example for inspection
        if last_example is not None:
            log_lines.append(f"last example from {spec['name']}:")
            log_lines.append(json.dumps(last_example, ensure_ascii=True, indent=2))
        print(f"finished dataset: {spec['name']}")

    train_dataset = datasets.concatenate_datasets(train_parts).shuffle(seed=42)
    test_dataset = datasets.concatenate_datasets(test_parts).shuffle(seed=42)

    # 16-gram deduplication between test and train (question text only),
    # applied only to non-MedCalcBench samples.
    print("starting 16-gram deduplication (test -> train)")
    medcalc_source = "ncbi/MedCalc-Bench"
    train_before = len(train_dataset)
    test_before = len(test_dataset)
    non_medcalc_test = test_dataset.filter(lambda x: x.get("data_source") != medcalc_source)
    test_ngrams = collect_ngrams_from_questions(non_medcalc_test, n=16)
    train_medcalc_before = sum(1 for x in train_dataset if x.get("data_source") == medcalc_source)
    train_non_medcalc_before = train_before - train_medcalc_before
    train_dataset = train_dataset.filter(
        lambda x: (
            x.get("data_source") == medcalc_source
            or not has_ngram_overlap(x.get("extra_info", {}).get("question"), test_ngrams, n=16)
        )
    )
    train_after = len(train_dataset)
    train_medcalc_after = sum(1 for x in train_dataset if x.get("data_source") == medcalc_source)
    train_non_medcalc_after = train_after - train_medcalc_after
    print("finished 16-gram deduplication")
    log_lines.append(
        "16-gram deduplication between test and train (question text only, excluding MedCalcBench)"
    )
    log_lines.append(f"train before: {train_before}")
    log_lines.append(f"train after: {train_after}")
    log_lines.append(f"train removed: {train_before - train_after}")
    log_lines.append(f"train non-medcalc before: {train_non_medcalc_before}")
    log_lines.append(f"train non-medcalc after: {train_non_medcalc_after}")
    log_lines.append(f"train non-medcalc removed: {train_non_medcalc_before - train_non_medcalc_after}")
    log_lines.append(f"train medcalc kept: {train_medcalc_after}")
    log_lines.append(f"test size: {test_before}")
    log_lines.append(f"test non-medcalc size used for n-grams: {len(non_medcalc_test)}")
    log_lines.append(f"test unique 16-grams: {len(test_ngrams)}")

    if args.train_limit:
        train_before_limit = len(train_dataset)
        train_dataset, train_kept_per_source = cap_dataset_per_source(train_dataset, 10000)
        log_lines.append("applied train per-source cap after 16-gram filtering: 10000")
        log_lines.append(f"train before cap: {train_before_limit}")
        log_lines.append(f"train after cap: {len(train_dataset)}")
        for source in sorted(train_kept_per_source.keys()):
            log_lines.append(f"train kept for {source}: {train_kept_per_source[source]}")

    if args.test_limit:
        test_before_limit = len(test_dataset)
        test_dataset, test_kept_per_source = cap_dataset_per_source(test_dataset, 1000)
        log_lines.append("applied test per-source cap after 16-gram filtering: 1000")
        log_lines.append(f"test before cap: {test_before_limit}")
        log_lines.append(f"test after cap: {len(test_dataset)}")
        for source in sorted(test_kept_per_source.keys()):
            log_lines.append(f"test kept for {source}: {test_kept_per_source[source]}")

    log_lines.append(f"final train rows: {len(train_dataset)}")
    log_lines.append(f"final valid rows: {len(test_dataset)}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))
