"""
Arguments:
    --data_path: path to MedMix test.parquet
    --answer_dir: result directory

Outputs:
    Writes per-sample json files: result_{file_stub}.json
    Writes aggregated stats jsonl: result.jsonl

Logs:
- /logs/search_{file_stub}.log: retrieval query and docs
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
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import pandas as pd

from search_agent.pre_retrieval_prompts import *
from search_agent.retrieval import query_clueweb, query_fineweb, query_medcorp, query_serper
from search_agent.utils import tokenize, query_qwen, query_gemini
from rewards.medmix_reward import compute_score


# Load environment variables from keys.env
load_dotenv(os.environ["KEYS_ENV_PATH"], override=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_ID = "gemini-2.5-flash"

CONCURRENT_NUM = 4


class RAGGenerator:
    def __init__(
        self,
        answer_dir: str,
        log_dir: str,
        model: str = "gemini",
        search_engine: str = "medcorp",
        num_docs: int = 3,
        url: str = "http://localhost:8000/v1",
        verbose: bool = False,
        result_jsonl: str | None = None,
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
        else:
            self.model_name = GEMINI_MODEL_ID
            self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.answer_dir = answer_dir
        self.log_dir = log_dir
        self.search_engine = search_engine
        self.num_docs = num_docs
        self.verbose = verbose
        self.result_jsonl = result_jsonl

        print(
            f"#######\nInit RAGGenerator with model {self.model_name}, search: {self.search_engine}, num_docs: {self.num_docs}\n#######"
        )
        time.sleep(2)

    def run_parallel(
        self,
        task_descriptions,
        questions,
        question_ids,
        prompt_ids,
        prompt_templates,
        file_stubs,
        data_sources,
        sample_ids,
        ground_truths,
        extra_infos,
    ):
        print(f"Running {len(task_descriptions)} questions in parallel with {CONCURRENT_NUM} workers")

        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = [
                executor.submit(
                    self.run_one,
                    task_description,
                    question,
                    question_id,
                    prompt_id,
                    prompt_template,
                    file_stub,
                    data_source,
                    sample_id,
                    ground_truth,
                    extra_info,
                )
                for task_description, question, question_id, prompt_id, prompt_template, file_stub, data_source, sample_id, ground_truth, extra_info in zip(
                    task_descriptions,
                    questions,
                    question_ids,
                    prompt_ids,
                    prompt_templates,
                    file_stubs,
                    data_sources,
                    sample_ids,
                    ground_truths,
                    extra_infos,
                )
            ]
            concurrent.futures.wait(futures)
            for future in futures:
                future.result()

    def run_one(
        self,
        task_description,
        question,
        question_id,
        prompt_id,
        prompt_template,
        file_stub,
        data_source,
        sample_id,
        ground_truth,
        extra_info,
    ):
        os.makedirs(self.log_dir, exist_ok=True)
        search_log = f"{self.log_dir}/search_{file_stub}.log"

        if self.verbose:
            with open(search_log, "w", encoding="utf-8") as f:
                f.write("")

        print(f"Running question {question_id}")
        try:
            retrieved_docs, _ = self._search(task_description, search_log, question_id)
            prompt = prompt_template.format(
                task_description=task_description,
                retrieved_docs=retrieved_docs,
            )

            if self.model == "qwen":
                thinking, response = self._query_qwen(prompt)
            else:
                thinking, response = self._query_gemini(prompt)

            response = (response or "").strip()
            thinking = (thinking or "").strip()
            prompt_length = tokenize(prompt)
            thinking_length = tokenize(thinking)
            response_length = tokenize(response)
            thinking_citation_count = self._count_citations(thinking)
            response_citation_count = self._count_citations(response)
            complete = response != ""
            self._log_result(
                answer=response,
                thinking=thinking,
                question_id=question_id,
                prompt_id=prompt_id,
                question=question,
                complete=complete,
                prompt_length=prompt_length,
                thinking_length=thinking_length,
                response_length=response_length,
                thinking_citation_count=thinking_citation_count,
                response_citation_count=response_citation_count,
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

    def _query_qwen(self, prompt):
        return query_qwen(self.client, self.model_name, prompt)

    def _query_gemini(self, prompt):
        return query_gemini(self.client, self.model_name, prompt, max_output_tokens=4096)

    def _extract_citation_tokens(self, text: str) -> list[str]:
        """Extract citation tokens from bracket groups, supporting [1][2] and [1,2]."""
        if not text:
            return []
        tokens = []
        for m in re.finditer(r"\[([^\[\]]+)\]", text):
            content = m.group(1).strip()
            if not content:
                continue
            for token in re.split(r"[\s,;/]+", content):
                if re.fullmatch(r"\d+", token) or re.fullmatch(r"T\d+-R\d+", token):
                    tokens.append(token)
        return tokens

    def _count_citations(self, text: str) -> int:
        return len(self._extract_citation_tokens(text))

    def _strip_answer_citations(self, text: str) -> str:
        if not text:
            return ""
        # Remove inline citation markers like [1], [1,2], [T2-R3], [1; T2-R3].
        def _remove_citation_group(match):
            content = match.group(1).strip()
            if not content:
                return match.group(0)
            parts = [p for p in re.split(r"[\s,;/]+", content) if p]
            if parts and all(re.fullmatch(r"\d+", p) or re.fullmatch(r"T\d+-R\d+", p) for p in parts):
                return ""
            return match.group(0)

        stripped = re.sub(r"\[([^\[\]]+)\]", _remove_citation_group, text)
        # Normalize extra spaces introduced by citation removal.
        stripped = re.sub(r"[ \t]{2,}", " ", stripped)
        stripped = re.sub(r" *\n", "\n", stripped)
        return stripped.strip()

    def _search(self, query, search_log, question_id):
        if self.search_engine == "clueweb":
            documents = query_clueweb(query, num_docs=self.num_docs)
        elif self.search_engine == "serper":
            documents = query_serper(query, num_docs=self.num_docs)
        elif self.search_engine == "medcorp":
            documents = query_medcorp(query, num_docs=self.num_docs)
        elif self.search_engine == "fineweb":
            documents = query_fineweb(query, num_docs=self.num_docs)
        else:
            raise ValueError(f"Invalid search engine: {self.search_engine}")

        formatted_documents = [f"[{idx}] {doc}" for idx, doc in enumerate(documents, start=1)]
        info_retrieved = "\n\n".join(formatted_documents)

        if self.verbose:
            with open(search_log, "a", encoding="utf-8") as f:
                f.write(f"[question_id={question_id}]\n")
                f.write(f"query:\n{query}\n\n")
                f.write(f"info_retrieved:\n{info_retrieved}\n\n\n")

        return info_retrieved, len(documents)

    def _log_result(
        self,
        answer,
        thinking,
        question_id,
        prompt_id,
        question,
        complete,
        prompt_length,
        thinking_length,
        response_length,
        thinking_citation_count,
        response_citation_count,
        file_stub,
        data_source,
        sample_id,
        ground_truth,
        extra_info,
    ):
        answer_no_citations = self._strip_answer_citations(answer)
        try:
            score = compute_score(
                data_source=data_source,
                solution_str=answer_no_citations,
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
                "thinking": thinking,
                "answer": answer,
                "answer_no_citations": answer_no_citations,
                "ground_truth": ground_truth,
                "pred_boxed_answer": boxed_answer,
                "score": score,
                "answer_length": answer_length,
                "extra_info": extra_info,
                "thinking citation count": thinking_citation_count,
                "response citation count": response_citation_count,
                "prompt length": prompt_length,
                "thinking length": thinking_length,
                "response length": response_length,
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
        prompt = row.get("prompt", "")
        if not isinstance(extra_info, dict):
            extra_info = {}
        if not isinstance(reward_model, dict):
            reward_model = {}

        sample_id = extract_sample_id(extra_info, row_idx)
        question = prompt
        task_description = extract_user_content(prompt)
        use_calc_template = str(data_source) == "ncbi/MedCalc-Bench"
        safe_data_source = sanitize_for_filename(data_source)
        safe_sample_id = sanitize_for_filename(sample_id)
        file_stub = f"{safe_data_source}_{safe_sample_id}"

        samples.append(
            {
                "question_id": row_idx,
                "prompt_id": row_idx,
                "file_stub": file_stub,
                "question": prompt,
                "task_description": task_description,
                "prompt_template": CALC_RAG_TEMPLATE if use_calc_template else QA_RAG_TEMPLATE,
                "data_source": data_source,
                "sample_id": sample_id,
                "ground_truth": reward_model.get("ground_truth"),
                "extra_info": extra_info,
            }
        )
    return samples


def filter_completed_samples(samples, answer_dir):
    """Filter out samples that already have answer files."""
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

            thinking_nonzero = sum(int(r.get("thinking citation count", 0) or 0) > 0 for r in records)
            response_nonzero = sum(int(r.get("response citation count", 0) or 0) > 0 for r in records)
            summary = {
                "data_source": data_source,
                "num_samples": n,
                "accuracy": sum(float(r.get("score", 0.0)) for r in records) / n,
                "complete_proportion": sum(bool(r.get("complete", False)) for r in records) / n,
                "thinking_citation_nonzero_proportion": thinking_nonzero / n,
                "response_citation_nonzero_proportion": response_nonzero / n,
                "mean_thinking_citation_count": sum(float(r.get("thinking citation count", 0.0)) for r in records) / n,
                "mean_response_citation_count": sum(float(r.get("response citation count", 0.0)) for r in records) / n,
                "mean_prompt_length": sum(float(r.get("prompt length", 0.0)) for r in records) / n,
                "mean_thinking_length": sum(float(r.get("thinking length", 0.0)) for r in records) / n,
                "mean_response_length": sum(float(r.get("response length", 0.0)) for r in records) / n,
                "mean_answer_length": sum(float(r.get("answer_length", 0.0)) for r in records) / n,
            }
            f.write(json.dumps(summary, ensure_ascii=True) + "\n")

        if all_records:
            n_all = len(all_records)
            thinking_nonzero_all = sum(int(r.get("thinking citation count", 0) or 0) > 0 for r in all_records)
            response_nonzero_all = sum(int(r.get("response citation count", 0) or 0) > 0 for r in all_records)
            overall = {
                "data_source": "__overall__",
                "num_samples": n_all,
                "accuracy": sum(float(r.get("score", 0.0)) for r in all_records) / n_all,
                "complete_proportion": sum(bool(r.get("complete", False)) for r in all_records) / n_all,
                "thinking_citation_nonzero_proportion": thinking_nonzero_all / n_all,
                "response_citation_nonzero_proportion": response_nonzero_all / n_all,
                "mean_thinking_citation_count": sum(float(r.get("thinking citation count", 0.0)) for r in all_records) / n_all,
                "mean_response_citation_count": sum(float(r.get("response citation count", 0.0)) for r in all_records) / n_all,
                "mean_prompt_length": sum(float(r.get("prompt length", 0.0)) for r in all_records) / n_all,
                "mean_thinking_length": sum(float(r.get("thinking length", 0.0)) for r in all_records) / n_all,
                "mean_response_length": sum(float(r.get("response length", 0.0)) for r in all_records) / n_all,
                "mean_answer_length": sum(float(r.get("answer_length", 0.0)) for r in all_records) / n_all,
            }
            f.write(json.dumps(overall, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/path/to/data/medmix/test.parquet", help="Path to MedMix test.parquet file")
    parser.add_argument("--log_dir", type=str, default="/path/to/RAG/gemini_2p5_flash_medmix/logs_clueweb", help="Log directory")
    parser.add_argument("--answer_dir", type=str, default="/path/to/RAG/gemini_2p5_flash_medmix/results_clueweb", help="Result directory")
    parser.add_argument("--model", default="gemini", help="base model")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="URL to use")
    parser.add_argument("--search_engine", type=str, default="clueweb", help="Search engine to use: clueweb | serper | medcorp | fineweb")
    parser.add_argument("--num_docs", type=int, default=5, help="Number of docs to retrieve")
    parser.add_argument("--result_jsonl", type=str, default=None, help="Path to write aggregated stats jsonl. Defaults to {answer_dir}/result.jsonl")
    parser.add_argument("--sample_limit", type=int, default=100, help="Maximum number of samples per data_source from test.parquet")
    parser.add_argument("--data_source", nargs="+", default=None, help="Only process specified data_source values")
    parser.add_argument("--debug", action="store_true", help="Process only first 10 samples after filtering")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    answer_dir = args.answer_dir
    log_dir = args.log_dir
    model = args.model
    url = args.url

    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    result_jsonl_path = args.result_jsonl or os.path.join(answer_dir, "result.jsonl")

    samples = load_samples_from_medmix_parquet(
        args.data_path,
        sample_limit=args.sample_limit,
        sample_seed=42,
        data_sources=args.data_source,
    )
    total_questions = len(samples)
    print(f"Loaded {total_questions} samples from {args.data_path}")

    remaining_samples, completed_count = filter_completed_samples(samples, answer_dir)
    if args.debug:
        remaining_samples = remaining_samples[:10]
    remaining_questions_num = len(remaining_samples)

    print(f"Total dataset: {total_questions} questions")
    print(f"Already completed: {completed_count} questions")
    print(f"Remaining to process: {remaining_questions_num} questions")

    if remaining_questions_num == 0:
        write_stats_result_jsonl(answer_dir, result_jsonl_path)
        print(f"Aggregated stats jsonl written to {result_jsonl_path}")
        print("All questions have been completed!")
        raise SystemExit(0)

    task_descriptions = [sample["task_description"] for sample in remaining_samples]
    remaining_questions = [sample["question"] for sample in remaining_samples]
    remaining_ids = [sample["question_id"] for sample in remaining_samples]
    remaining_prompt_ids = [sample["prompt_id"] for sample in remaining_samples]
    remaining_prompt_templates = [sample["prompt_template"] for sample in remaining_samples]
    remaining_file_stubs = [sample["file_stub"] for sample in remaining_samples]
    remaining_data_sources = [sample["data_source"] for sample in remaining_samples]
    remaining_sample_ids = [sample["sample_id"] for sample in remaining_samples]
    remaining_ground_truths = [sample["ground_truth"] for sample in remaining_samples]
    remaining_extra_infos = [sample["extra_info"] for sample in remaining_samples]

    generator = RAGGenerator(
        answer_dir=answer_dir,
        log_dir=log_dir,
        model=model,
        search_engine=args.search_engine,
        num_docs=args.num_docs,
        url=url,
        verbose=True,
        result_jsonl=result_jsonl_path,
    )

    generator.run_parallel(
        task_descriptions,
        remaining_questions,
        remaining_ids,
        remaining_prompt_ids,
        remaining_prompt_templates,
        remaining_file_stubs,
        remaining_data_sources,
        remaining_sample_ids,
        remaining_ground_truths,
        remaining_extra_infos,
    )
    write_stats_result_jsonl(answer_dir, result_jsonl_path)
    print(f"Aggregated stats jsonl written to {result_jsonl_path}")
