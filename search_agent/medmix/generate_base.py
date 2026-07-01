"""
Arguments:
    --data_path: path to MedMix test.parquet
    --answer_dir: result directory

Outputs:
    Writes per-sample json files: result_{data_source}_{sample_id}.json
    Writes aggregated stats jsonl: result.jsonl
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as generativeai
import pandas as pd
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import boto3
from botocore.config import Config

from search_agent.agent_search_prompts import CALC_BASE_TEMPLATE, QA_BASE_TEMPLATE
from search_agent.utils import tokenize, query_qwen, query_gemini, query_gpt, query_bedrock
from rewards.medmix_reward import compute_score


# Load environment variables from keys.env
load_dotenv(os.environ["KEYS_ENV_PATH"], override=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_ID = "gemini-2.5-flash"

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.deepseek.r1-v1:0")

_BEDROCK_CONFIG = Config(
    region_name=os.getenv("BEDROCK_REGION", "us-east-2"),
    retries={"max_attempts": 3, "mode": "standard"},
    read_timeout=600,
    connect_timeout=10,
    max_pool_connections=128,
)

_bedrock_client = None

def get_bedrock_client():
    """Return a cached Bedrock runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", config=_BEDROCK_CONFIG)
    return _bedrock_client

CONCURRENT_NUM = 8


def _strip_think(text: str | None) -> str:
    if not text:
        return ""
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return stripped.strip()


class BaseGenerator:
    def __init__(
        self,
        answer_dir: str,
        model: str = "gemini",
        url: str = "http://localhost:8000/v1",
        max_turns: int = 15,
    ):
        self.model = model
        if model in ["qwen", "gpt"]:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                base_url=None if model == "gpt" else url,
                timeout=60,
            )
            if model == "gpt":
                self.model_name = "gpt-5-mini"
            else:
                self.model_name = self.client.models.list().data[0].id
        elif model == "bedrock":
            self.client = get_bedrock_client()
            self.model_name = BEDROCK_MODEL_ID
        else:
            self.model_name = GEMINI_MODEL_ID
            self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.answer_dir = answer_dir
        self.max_turns = max_turns
        print(f"#######\nInit BaseGenerator with model {self.model_name}\n#######")

    def run_parallel(
        self,
        prompts,
        questions,
        question_ids,
        prompt_ids,
        file_stubs,
        data_sources,
        sample_ids,
        ground_truths,
        extra_infos,
    ):
        print(f"Running {len(prompts)} questions in parallel with {CONCURRENT_NUM} workers")

        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = [
                executor.submit(
                    self.run_one,
                    prompt,
                    question,
                    question_id,
                    prompt_id,
                    file_stub,
                    data_source,
                    sample_id,
                    ground_truth,
                    extra_info,
                )
                for prompt, question, question_id, prompt_id, file_stub, data_source, sample_id, ground_truth, extra_info in zip(
                    prompts,
                    questions,
                    question_ids,
                    prompt_ids,
                    file_stubs,
                    data_sources,
                    sample_ids,
                    ground_truths,
                    extra_infos,
                )
            ]
            concurrent.futures.wait(futures)

    def run_one(
        self,
        prompt,
        question,
        question_id,
        prompt_id,
        file_stub,
        data_source,
        sample_id,
        ground_truth,
        extra_info,
    ):
        print(f"Running question {question_id}")
        try:
            answer = ""
            complete = False
            for _ in range(self.max_turns):
                if self.model == "qwen":
                    response = self._query_qwen(prompt)
                elif self.model == "gpt":
                    response = self._query_gpt(prompt)
                elif self.model == "bedrock":
                    response = self._query_bedrock(prompt)
                else:
                    response = self._query_gemini(prompt)

                answer, complete = self._compose_final_output(response)
                if complete:
                    break

            self._log_result(
                answer=answer,
                question_id=question_id,
                prompt_id=prompt_id,
                question=question,
                complete=complete,
                file_stub=file_stub,
                data_source=data_source,
                sample_id=sample_id,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            print(f"Question {question_id} result saved to {self.answer_dir}/result_{file_stub}.json\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def _compose_final_output(self, response):
        cleaned = _strip_think(response)
        start = "<response>"
        end = "</response>"
        if start in cleaned and end in cleaned:
            return cleaned.split(start, 1)[1].split(end, 1)[0].strip(), True
        if re.search(r"\\boxed\{.*?\}", cleaned, flags=re.DOTALL):
            return cleaned, True
        return cleaned, False

    def _query_qwen(self, prompt):
        _, response = query_qwen(self.client, self.model_name, prompt)
        return response

    def _query_gpt(self, prompt):
        _, response = query_gpt(self.client, self.model_name, prompt)
        return response

    def _query_gemini(self, prompt):
        _, response = query_gemini(self.client, self.model_name, prompt, with_thinking=False)
        return response

    def _query_bedrock(self, prompt):
        _, response = query_bedrock(self.client, self.model_name, prompt)
        return response

    def _log_result(
        self,
        answer,
        question_id,
        prompt_id,
        question,
        complete,
        file_stub,
        data_source,
        sample_id,
        ground_truth,
        extra_info,
    ):
        try:
            score = compute_score(
                data_source=data_source,
                solution_str=answer,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        except Exception:
            score = 0.0
        answer_length = tokenize(answer)
        boxed_answer = extract_last_boxed_content(answer)
        if hasattr(question, "tolist"):
            question = question.tolist()
        answer_file = f"{self.answer_dir}/result_{file_stub}.json"
        with open(answer_file, "w", encoding="utf-8") as f:
            result = {
                "model": self.model_name,
                "data_source": data_source,
                "id": sample_id,
                "question_id": question_id,
                "question": question,
                "prompt_id": prompt_id,
                "complete": complete,
                "answer": answer,
                "ground_truth": ground_truth,
                "pred_boxed_answer": boxed_answer,
                "score": score,
                "answer_length": answer_length,
                "extra_info": extra_info,
            }
            json.dump(result, f, indent=4)


def sanitize_for_filename(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def extract_last_boxed_content(text):
    matches = re.findall(r"\\boxed\{(.*?)\}", text or "", flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def extract_user_content(prompt):
    messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    if isinstance(messages, dict):
        messages = [messages]
    for msg in messages:
        if str(msg["role"]).lower() == "user":
            return str(msg["content"])
    return ""


def extract_sample_id(extra_info, fallback_id):
    if isinstance(extra_info, dict):
        for key in ["id", "qid", "question_id", "row_number", "index"]:
            if key in extra_info and extra_info[key] is not None:
                return extra_info[key]
    return fallback_id


def load_samples_from_medmix_parquet(file_path, sample_limit=500, sample_seed=42, data_sources=None):
    df = pd.read_parquet(file_path)
    if data_sources:
        requested_sources = {str(source) for source in data_sources}
        df = df[df["data_source"].astype(str).isin(requested_sources)]
    if sample_limit is not None and sample_limit > 0:
        sampled_parts = []
        for _, source_df in df.groupby("data_source", dropna=False, sort=False):
            if len(source_df) > sample_limit:
                sampled_parts.append(source_df.sample(n=sample_limit, random_state=sample_seed))
            else:
                sampled_parts.append(source_df)
        if sampled_parts:
            df = pd.concat(sampled_parts, axis=0, ignore_index=True)
    df = df.reset_index(drop=True)

    samples = []
    for row_idx, row in df.iterrows():
        data_source = row.get("data_source", "unknown")
        extra_info = row.get("extra_info", {})
        reward_model = row.get("reward_model", {})
        prompt = row.get("prompt")

        if not isinstance(extra_info, dict):
            extra_info = {}
        if not isinstance(reward_model, dict):
            reward_model = {}

        sample_id = extra_info["qid"]
        user_content = extract_user_content(prompt)
        use_calc_template = str(data_source) == "ncbi/MedCalc-Bench"
        safe_data_source = sanitize_for_filename(data_source)
        safe_sample_id = sanitize_for_filename(sample_id)
        file_stub = f"{safe_data_source}_{safe_sample_id}"

        samples.append(
            {
                "question_id": row_idx,
                "prompt_id": row_idx,
                "data_source": data_source,
                "sample_id": sample_id,
                "file_stub": file_stub,
                "question": prompt,
                "task_description": user_content,
                "base_template": CALC_BASE_TEMPLATE if use_calc_template else QA_BASE_TEMPLATE,
                "ground_truth": reward_model.get("ground_truth"),
                "extra_info": extra_info,
            }
        )
    return samples


def filter_completed_samples(samples, answer_dir):
    """Filter out questions that already have answer files."""
    filtered_samples = []
    completed_count = 0

    for sample in samples:
        answer_file = f"{answer_dir}/result_{sample['file_stub']}.json"
        if os.path.exists(answer_file):
            completed_count += 1
        else:
            filtered_samples.append(sample)

    return filtered_samples, completed_count


def write_stats_result_jsonl(answer_dir, result_jsonl_path):
    result_files = sorted(glob.glob(os.path.join(answer_dir, "result_*.json")))
    grouped = {}
    for file_path in result_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            continue
        data_source = str(record.get("data_source", "unknown"))
        grouped.setdefault(data_source, []).append(record)

    with open(result_jsonl_path, "w", encoding="utf-8") as f:
        all_records = []
        for data_source in sorted(grouped.keys()):
            records = grouped[data_source]
            all_records.extend(records)
            n = len(records)
            if n == 0:
                continue
            summary = {
                "data_source": data_source,
                "num_samples": n,
                "score": sum(float(r.get("score", 0.0)) for r in records) / n,
                "answer_length": sum(float(r.get("answer_length", 0.0)) for r in records) / n,
            }
            f.write(json.dumps(summary, ensure_ascii=True) + "\n")

        if all_records:
            n_all = len(all_records)
            overall = {
                "data_source": "__overall__",
                "num_samples": n_all,
                "score": sum(float(r.get("score", 0.0)) for r in all_records) / n_all,
                "answer_length": sum(float(r.get("answer_length", 0.0)) for r in all_records) / n_all,
            }
            f.write(json.dumps(overall, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/data/medmix/test.parquet', help="Path to MedMix test.parquet file")
    parser.add_argument("--answer_dir", type=str, default="/path/to/output/MedAgent/deepseek_r1_medmix_citation/results_base", help="Result directory")
    parser.add_argument("--model", default="bedrock", help="base model: gemini | qwen | gpt | bedrock")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="URL to use")
    parser.add_argument("--result_jsonl", type=str, default=None, help="Path to write aggregated stats jsonl. Defaults to {answer_dir}/result.jsonl")
    parser.add_argument("--sample_limit", type=int, default=10, help="Maximum number of samples per data_source")
    parser.add_argument("--data_source", nargs='+', default=None, help="Only process specified data_source values. Supports multiple values")
    parser.add_argument("--max_turns", type=int, default=5, help="Max number of turns")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    answer_dir = args.answer_dir
    model = args.model
    url = args.url
    result_jsonl_path = args.result_jsonl or os.path.join(answer_dir, "result.jsonl")

    os.makedirs(answer_dir, exist_ok=True)

    samples = load_samples_from_medmix_parquet(
        args.data_path,
        sample_limit=args.sample_limit,
        sample_seed=42,
        data_sources=args.data_source,
    )

    total_questions = len(samples)
    print(f"Loaded {total_questions} samples from {args.data_path}")

    remaining_samples, completed_count = filter_completed_samples(samples, answer_dir)
    remaining_questions_num = len(remaining_samples)

    print(f"Total dataset: {total_questions} questions")
    print(f"Already completed: {completed_count} questions")
    print(f"Remaining to process: {remaining_questions_num} questions")

    if remaining_questions_num == 0:
        write_stats_result_jsonl(answer_dir, result_jsonl_path)
        print(f"Aggregated stats jsonl written to {result_jsonl_path}")
        print("All questions have been completed!")
        raise SystemExit(0)

    prompts = []
    remaining_questions = []
    remaining_ids = []
    remaining_prompt_ids = []
    remaining_file_stubs = []
    remaining_data_sources = []
    remaining_sample_ids = []
    remaining_ground_truths = []
    remaining_extra_infos = []

    for sample in remaining_samples:
        remaining_questions.append(sample["question"])
        remaining_ids.append(sample["question_id"])
        remaining_prompt_ids.append(sample["prompt_id"])
        remaining_file_stubs.append(sample["file_stub"])
        remaining_data_sources.append(sample["data_source"])
        remaining_sample_ids.append(sample["sample_id"])
        remaining_ground_truths.append(sample["ground_truth"])
        remaining_extra_infos.append(sample["extra_info"])

        prompts.append(
            sample["base_template"].format(task_description=sample["task_description"])
        )

    generator = BaseGenerator(
        answer_dir=answer_dir,
        model=model,
        url=url,
        max_turns=args.max_turns,
    )

    generator.run_parallel(
        prompts,
        remaining_questions,
        remaining_ids,
        remaining_prompt_ids,
        remaining_file_stubs,
        remaining_data_sources,
        remaining_sample_ids,
        remaining_ground_truths,
        remaining_extra_infos,
    )
    write_stats_result_jsonl(answer_dir, result_jsonl_path)
    print(f"Aggregated stats jsonl written to {result_jsonl_path}")
