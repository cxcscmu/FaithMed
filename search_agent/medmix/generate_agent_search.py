'''
Arguments:
    --data_path: path to MedMix test.parquet
    --answer_dir: result directory
    --log_dir: log directory

Logs:
- see all system messages in /logs
    - /logs/search_{data_source}_{id}.log: search query and results
    - /logs/trajectory_{data_source}_{id}.md: total trajectory of the agent

'''

import re, os
import argparse
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from search_agent.retrieval import *
from search_agent.factual_citation_prompts import *
import json
import traceback
import glob
from search_agent.utils import tokenize, query_qwen, query_qwen_local, query_gemini, query_bedrock
import pandas as pd
from rewards.medmix_reward import compute_score
from tqdm import tqdm

# Load environment variables from keys.env file (optional — may not exist for local-only runs)
_keys_env_path = os.environ.get("KEYS_ENV_PATH")
if _keys_env_path:
    load_dotenv(_keys_env_path, override=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash"

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.deepseek.r1-v1:0")

_bedrock_client = None

def get_bedrock_client():
    """Return a cached Bedrock runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        from botocore.config import Config
        _BEDROCK_CONFIG = Config(
            region_name=os.getenv("BEDROCK_REGION", "us-east-2"),
            retries={"max_attempts": 8, "mode": "standard"},
            read_timeout=120,
            connect_timeout=10,
            max_pool_connections=128,
        )
        _bedrock_client = boto3.client("bedrock-runtime", config=_BEDROCK_CONFIG)
    return _bedrock_client


ACTIONS = ['search', 'response', 'summary']
CONCURRENT_NUM = 32
MAX_CONTEXT_LENGTH = 8000


class LLMAgent:
    def __init__(
        self,
        config,
        log_dir: str,
        answer_dir: str,
        verbose: bool = False,
        model='gemini',
        search_engine: str = 'clueweb',
        url: str = 'http://localhost:8000/v1',
        result_jsonl: str | None = None,
        temperature: float = 0.0,
        model_path: str = '',
    ):
        self.model = model
        self.temperature = temperature
        if model in ['qwen', 'gpt']:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                base_url=None if model == 'gpt' else url,
                timeout=60
            )
            if model == 'gpt':
                self.model_name = "gpt-5-mini"
            else:
                self.model_name = self.client.models.list().data[0].id
        elif model == 'qwen_local':
            from vllm import LLM
            from transformers import AutoTokenizer as _AutoTokenizer
            self.local_tokenizer = _AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.llm = LLM(model=model_path, trust_remote_code=True)
            self.model_name = model_path
            self.client = None
        elif model == 'bedrock':
            self.client = get_bedrock_client()
            self.model_name = BEDROCK_MODEL_ID
        else:
            from google import genai
            self.model_name = GEMINI_MODEL_ID
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.search_engine = search_engine
        self.consecutive_search_cnt = {} # Number of consecutive search actions performed for each sample
        self.search_cnt = {} # Number of total search actions performed for each sample
        self.context_cnt = {} # Number of total context length in each turn for each sample
        self.information_cnt = {} # Number of information length in each search turn for each sample
        self.summary_cnt = {} # Number of summary actions performed for each sample
        self.turn_id = {} # Turn ID for each question
        self.summary_history = {} # Latest summary content for each sample
        self.need_format_reminder = {} # Whether format reminder is needed for each sample
        self.retrieved_doc_index = {} # Cached docs by (turn_id, result_idx), preserved even after summary compression
        self.config = config
        self.verbose = verbose
        self.log_dir = log_dir
        self.answer_dir = answer_dir
        self.questions = {}
        self.prompt_ids = {}
        self.memory_context = {}
        self.sample_meta = {}
        self.result_jsonl = result_jsonl
        self.result_jsonl_lock = Lock() if result_jsonl else None
        print(f"#######\nInit LLMAgent with model {self.model_name}, search: {self.search_engine}\n#######")
        time.sleep(5)


    def run_llm_loop(self, task_description, question_id):
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        meta = self.sample_meta[question_id]
        file_stub = meta["file_stub"]
        trajectory_log = f"{self.log_dir}/trajectory_{file_stub}.md"
        trajectory_jsonl_log = f"{self.log_dir}/trajectory_{file_stub}.jsonl"
        search_log = f"{self.log_dir}/search_{file_stub}.log"
        
        # Clear logs
        with open(trajectory_log, 'w', encoding='utf-8') as f:
            f.write('')
        if self.verbose:
            # with open(search_log, 'w', encoding='utf-8') as f:
            #     f.write('')
            with open(trajectory_jsonl_log, 'w', encoding='utf-8') as f:
                f.write('')

        print(f"Running question {question_id}")

        done = False
        input = meta["template_no_info"].format(
            task_description=task_description,
            memory_context="",
        )
        self.consecutive_search_cnt[question_id] = 0
        self.search_cnt[question_id] = 0
        self.context_cnt[question_id] = []
        self.information_cnt[question_id] = []
        self.summary_cnt[question_id] = 0
        self.turn_id[question_id] = 0
        self.memory_context[question_id] = []
        self.summary_history[question_id] = ""
        self.need_format_reminder[question_id] = False
        self.retrieved_doc_index[question_id] = {}
        try:
            for _ in range(self.config["max_turns"]):
                self.turn_id[question_id] += 1
                print(f"=====turn {self.turn_id[question_id]}======")

                if self.model == 'qwen':
                    response, processed_response = self._query_qwen(input, question_id)
                elif self.model == 'qwen_local':
                    response, processed_response = self._query_qwen_local(input, question_id)
                elif self.model == 'bedrock':
                    response, processed_response = self._query_bedrock(input, question_id)
                else:
                    response, processed_response = self._query_gemini(input, question_id)
                # execute actions (search or response) and get observations
                done, need_update_history, next_obs = self._execute_response(
                    processed_response, self.config["num_docs"], question_id, search_log, raw_response=response
                )
                self._record_trajectory(input, response, next_obs, trajectory_log, trajectory_jsonl_log, question_id)

                if done:
                    print("=====final response======")
                    break
                input = self._update_input(
                    input,
                    response,
                    next_obs,
                    question_id,
                    task_description,
                    need_update_history,
                )
                # input = self._update_input( input, processed_response, next_obs, question_id, task_description)
            
            complete = self.check_completeness(processed_response)
            self._log_result(
                answer=processed_response,
                raw_response=response,
                question_id=question_id,
                prompt_id=self.prompt_ids.get(question_id, question_id),
                complete=complete,
            )
                
            print(f"Question {question_id} result saved to {self.answer_dir}/result_{file_stub}.json\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def run_llm_loop_parallel(self, samples):
        print(f"Running {len(samples)} questions in parallel with {CONCURRENT_NUM} workers")

        for sample in samples:
            question_id = sample["question_id"]
            self.questions[question_id] = sample["question"]
            self.prompt_ids[question_id] = sample["prompt_id"]
            self.sample_meta[question_id] = sample

        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = [
                executor.submit(self.run_llm_loop, sample["task_description"], sample["question_id"])
                for sample in samples
            ]
            with tqdm(total=len(futures), desc="Questions completed", unit="q") as pbar:
                for future in as_completed(futures):
                    future.result()  # Surface any setup-time failures from worker threads
                    pbar.update(1)

    def _query_qwen(self, prompt, question_id):
        thought, response = query_qwen(self.client, self.model_name, prompt, temperature=self.temperature)
        processed_response = self._postprocess_response(response)
        original_response = f'<think>{thought}</think>\n{response}' if thought else response
        self.context_cnt[question_id].append(tokenize(prompt) + tokenize(original_response))
        return original_response, processed_response

    def _query_qwen_local(self, prompt, question_id):
        thought, response = query_qwen_local(self.llm, self.local_tokenizer, prompt, temperature=self.temperature)
        processed_response = self._postprocess_response(response)
        original_response = f'<think>{thought}</think>\n{response}' if thought else response
        self.context_cnt[question_id].append(tokenize(prompt) + tokenize(original_response))
        return original_response, processed_response

    def _query_gemini(self, prompt, question_id):
        processed_response = None
        thought, response = "", ""
        for _ in range(5):
            thought, response = query_gemini(self.client, self.model_name, prompt, max_output_tokens=6144)
            if response:
                processed_response = self._postprocess_response(response)
                if processed_response is not None:
                    break
        original_response = f'<think>{thought}</think>\n{response}' if thought else response
        self.context_cnt[question_id].append(tokenize(prompt) + tokenize(original_response))
        return original_response, processed_response

    def _query_bedrock(self, prompt, question_id):
        processed_response = None
        thought, response = "", ""
        for _ in range(5):
            thought, response = query_bedrock(self.client, self.model_name, prompt, temperature=self.temperature)
            if response:
                processed_response = self._postprocess_response(response)
                if processed_response is not None:
                    break
        original_response = f"<think>{thought}</think>\n{response}" if thought else response
        self.context_cnt[question_id].append(tokenize(prompt) + tokenize(original_response))
        return original_response, processed_response

    def _postprocess_response(self, response: str):
        if response is None:
            return None

        # The model must never emit tool observations itself.
        if "<information>" in response or "</information>" in response:
            return None

        # Special handling for summary because summary content may include nested tags.
        if "<summary>" in response or "</summary>" in response:
            if response.count("<summary>") != 1 or response.count("</summary>") != 1:
                return None
            start_idx = response.find("<summary>")
            end_idx = response.rfind("</summary>")
            prefix = response[:start_idx].strip()
            suffix = response[end_idx + len("</summary>"):].strip()
            if prefix or suffix:
                return None
            content = response[start_idx + len("<summary>"):end_idx].strip()
            return f"<summary>{content}</summary>"

        m_search = re.search(r"<search>(.*?)</search>", response, flags=re.DOTALL)
        m_answer = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)

        # If both action blocks exist in one turn, treat as invalid.
        if m_search and m_answer:
            return None
    
        # return search query if search block exists
        if m_search:
            if response.strip() != m_search.group(0).strip():
                return None
            return m_search.group(0)

        if m_answer:
            prefix = response[:m_answer.start()].strip()
            content = m_answer.group(1)
            tail = response[m_answer.end():]
            if prefix:
                return None
            # if there is a boxed answer in the tail after </answer>, append it to the content of answer block
            m_boxed_tail = re.search(r"\\boxed\{.*?\}", tail, flags=re.DOTALL)
            tail_without_boxed = tail
            if m_boxed_tail and re.search(r"\\boxed\{.*?\}", content, flags=re.DOTALL) is None:
                content = content.rstrip() + "\n" + m_boxed_tail.group(0)
                tail_without_boxed = tail.replace(m_boxed_tail.group(0), "", 1)
            if tail_without_boxed.strip():
                return None
            if re.search(r"\\boxed\{.*?\}", content, flags=re.DOTALL) is None:
                return None
            return f"<answer>{content}</answer>"

        # fallback: if no answer block exists, but there is a boxed answer in the whole text, return the whole text as answer
        if re.search(r"\\boxed\{.*?\}", response, flags=re.DOTALL):
            return f"<answer>{response.strip()}</answer>"

        return None


    def _contains_fabricated_information_after_search(self, response):
        if not isinstance(response, str):
            return False
        response_wo_think = re.sub(r"^\s*<think>.*?</think>\s*", "", response, flags=re.DOTALL)
        return (
            "<search>" in response_wo_think
            and "</search>" in response_wo_think
            and "<information>" in response_wo_think
            and "</information>" in response_wo_think
        )

    def _execute_response(self, processed_response, num_docs, question_id, search_log, do_search=True, raw_response=None):
        """
        Args:
            action: action to be executed, None if format is not correct
            num_docs: number of docs to retrieve
            search_log: file to log search output
            do_search: whether to perform search
        Returns:
            done: whether the task is done
            need_update_history: whether summary action requests history compression
            next_obs: next observation
        """

        if processed_response is None:
            self.need_format_reminder[question_id] = True
            if self._contains_fabricated_information_after_search(raw_response):
                next_obs = (
                    'Invalid action: your output included a <search> action followed by a self-generated '
                    '<information> block. Do not generate <information> yourself; it is added by the system '
                    'only after a valid <search> action.'
                )
            else:
                next_obs = 'Invalid action, cannot be executed.'
            return False, False, next_obs

        action, content = self._parse_action(processed_response)
        next_obs = ''
        done = False
        need_update_history = False

        search_query = content if action == 'search' else ''
        
        if do_search and search_query != '':
            search_results, indexed_docs = self._search(search_query, num_docs, search_log, question_id)
        else:
            search_results = ''
            indexed_docs = {}

        if action == "answer":
            done = True
            next_obs = 'answer generated, the process is done.'
        elif action == 'search':
            self.search_cnt[question_id] += 1
            self.consecutive_search_cnt[question_id] += 1
            self.information_cnt[question_id].append(tokenize(search_results))
            for result_idx, doc_text in indexed_docs.items():
                self.retrieved_doc_index[question_id][(self.turn_id[question_id], result_idx)] = doc_text
            observation = f'<information>{search_results}</information>'
            next_obs = observation
        elif action == 'summary':
            self.consecutive_search_cnt[question_id] = 0
            self.summary_cnt[question_id] += 1
            self.summary_history[question_id] = content
            need_update_history = True
            next_obs = (
                'You performed a summary action in this turn. '
                'Your history will be compressed according to this summary.'
            )
        else:
            raise ValueError(f"Invalid action: {action}")

        return done, need_update_history, next_obs

    def _parse_action(self, processed_response):
        """Parse the action to get the action type and content.
        Args:
            processed_response: processed response, format ensured by postprocess_response
        Returns:
            action_type: action type
            content: action content
        """
        # Strip a leading <think>...</think> if present, to reach the action.
        action = re.sub(r"^\s*<think>.*?</think>\s*", "", processed_response, flags=re.DOTALL)
        # Find the first occurrence of '<' and '>' to extract action_type
        start_tag_open = action.find('<')
        start_tag_close = action.find('>', start_tag_open)
        if start_tag_open == -1 or start_tag_close == -1:
            raise ValueError(f"Invalid action format: {action}")
        
        action_type = action[start_tag_open + 1:start_tag_close]

        # Find the last occurrence of '</' and '>' to locate the closing tag
        end_tag_open = action.rfind('</')
        end_tag_close = action.rfind('>', end_tag_open)
        if end_tag_open == -1 or end_tag_close == -1:
            raise ValueError(f"Invalid action format: {action}")

        # Extract content between the first '>' and last '</'
        content = action[start_tag_close + 1:end_tag_open].strip()

        return action_type, content

    def _record_trajectory(self, input, response, next_obs, trajectory_log, trajectory_jsonl_log, question_id):
        """Record the trajectory of the agent.
        Args:
            input: input
            response: response
            trajectory_log: path to trajectory log file
        """
        def truncate_information_block(match):
            content = match.group(1).strip()
            pattern = r"(?ms)^Result \[T\d+-R\d+\]:\n.*?(?=^Result \[T\d+-R\d+\]:|\Z)"
            blocks = re.findall(pattern, content)
            if not blocks:
                return f"<information>{self._truncate_text(content)}</information>"

            truncated_blocks = []
            for block in blocks:
                header, sep, body = block.partition("\n")
                truncated_blocks.append(f"{header}\n{self._truncate_text(body.strip())}")
            return "<information>" + "\n\n".join(truncated_blocks) + "</information>"

        def truncate_generic_block(action, text):
            pattern = f'<{action}>(.*?)</{action}>'

            def _truncate(match):
                full_content = match.group(1)
                return f'<{action}>{self._truncate_text(full_content)}</{action}>'

            return re.sub(pattern, _truncate, text, flags=re.DOTALL)

        with open(trajectory_log, 'a', encoding='utf-8') as f:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"## Turn {self.turn_id[question_id]} {time}\n\n")

            input_length = tokenize(input)
            response_length = tokenize(response)

            input_short = input
            for action in ['search', 'answer', 'summary']:
                input_short = truncate_generic_block(action, input_short)
            input_short = re.sub(
                r'<information>(.*?)</information>',
                truncate_information_block,
                input_short,
                flags=re.DOTALL,
            )

            f.write(f"### Input:\n**length={input_length}**\n{input_short}\n\n")
            f.write(f"### Response:\n**length={response_length}**\n{response}\n\n--------------------------------\n\n")

        if self.verbose:
            with open(trajectory_jsonl_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "input": input,
                    "response": response,
                    "next_obs": next_obs,
                    "context_length": input_length + response_length
                }) + '\n')

    def _update_input(self, input, cur_response, next_obs, question_id, task_description, need_update_history=False):
        """Update the input with the history.
        Args:
            input: input
            cur_response: current full response
            next_obs: next observation
            task_description: original task description for the question
            need_update_history: whether history should be rewritten from summary action
        Returns:
            updated input
        """
        if self.need_format_reminder[question_id]:
            if self._contains_fabricated_information_after_search(cur_response):
                context = f"[Turn {self.turn_id[question_id]}]:\n{next_obs}\n\n"
            else:
                context = f"[Turn {self.turn_id[question_id]}]:\n{cur_response}\n\n"
            context += FORMAT_REMINDER_PROMPT
            new_input = input + context
            self.need_format_reminder[question_id] = False
        elif need_update_history:
            context = f"[Turn 0 - Turn {self.turn_id[question_id] - 1}]:\n{self.summary_history[question_id]}\n\n"
            context += f"[Turn {self.turn_id[question_id]}]:\n{next_obs}\n"
            self.memory_context[question_id] = [context]
            memory_context = "\n".join(self.memory_context[question_id]).strip()
            meta = self.sample_meta[question_id]
            template_key = "template_with_info" if self.search_cnt[question_id] > 0 else "template_no_info"
            new_input = meta[template_key].format(
                task_description=task_description,
                memory_context=memory_context,
            )
        else:
            context = f"[Turn {self.turn_id[question_id]}]:\n{cur_response}\n{next_obs}\n"
            self.memory_context[question_id].append(context)
            memory_context = "\n".join(self.memory_context[question_id]).strip()
            meta = self.sample_meta[question_id]
            template_key = "template_with_info" if self.search_cnt[question_id] > 0 else "template_no_info"
            new_input = meta[template_key].format(
                task_description=task_description,
                memory_context=memory_context,
            )
        if tokenize(new_input) > MAX_CONTEXT_LENGTH:
            new_input += SUMMARY_REMINDER_PROMPT
        return new_input
    
    def check_completeness(self, response):
        start = "<answer>"
        end = "</answer>"
        # If answer block exists and boxed answer is inside, it's complete.
        if start in response and end in response:
            answer = response.split(start, 1)[1].split(end, 1)[0]
            return re.search(r"\\boxed\{.*?\}", answer, flags=re.DOTALL) is not None
        return False

    def _index_docs_from_search_results(self, search_results):
        """Parse numbered search results into {result_idx: full_result_text}."""
        if not isinstance(search_results, str) or not search_results.strip():
            return {}
        text = search_results.strip()
        matches = list(re.finditer(r"(?ms)^\s*(\d+)\.\s+.*?(?=^\s*\d+\.\s+|\Z)", text))
        if not matches:
            return {1: text}

        out = {}
        for m in matches:
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            out[idx] = m.group(0).strip()
        return out

    def _extract_citations(self, text):
        """Return unique citations in appearance order, e.g. [(2,3), (4,1)].
        Handles [T1-R2], (T1-R2), and multi-citation groups like [T1-R2, T1-R3]."""
        if not isinstance(text, str):
            return []
        seen = set()
        ordered = []
        for match in re.finditer(r"[\[(]([^\[\]()]+)[\])]", text):
            for token in re.split(r"[\s,;/]+", match.group(1).strip()):
                m = re.fullmatch(r"T(\d+)-R(\d+)", token)
                if m:
                    key = (int(m.group(1)), int(m.group(2)))
                    if key not in seen:
                        seen.add(key)
                        ordered.append(key)
        return ordered

    def _count_resolved_citations(self, text, question_id):
        doc_map = self.retrieved_doc_index.get(question_id, {})
        return sum(
            1
            for turn_id, result_idx in self._extract_citations(text)
            if (turn_id, result_idx) in doc_map
        )

    def _extract_thinking_content(self, raw_response):
        if not isinstance(raw_response, str):
            return ""
        match = re.search(r"<think>(.*?)</think>", raw_response, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    def _truncate_text(self, text, max_chars=100):
        if not isinstance(text, str):
            return text
        return text if len(text) <= max_chars else text[:max_chars] + "..."

    def _format_retrieved_documents(self, documents, turn_id):
        indexed_docs = {}
        formatted_blocks = []
        for result_idx, doc_text in enumerate(documents, start=1):
            clean_text = (doc_text or "").strip()
            indexed_docs[result_idx] = clean_text
            formatted_blocks.append(
                f"Result [T{turn_id}-R{result_idx}]:\n{clean_text}"
            )
        return "\n\n".join(formatted_blocks), indexed_docs

    def _build_reference_entries(self, text, question_id):
        citations = self._extract_citations(text)
        doc_map = self.retrieved_doc_index.get(question_id, {})
        references = []
        for turn_id, result_idx in citations:
            tag = f"[T{turn_id}-R{result_idx}]"
            doc_text = doc_map.get((turn_id, result_idx))
            references.append(
                {
                    "tag": tag,
                    "document": (
                        self._truncate_text(doc_text)
                        if doc_text
                        else "[corresponding document not found]"
                    ),
                }
            )
        return references

    def _log_result(self, answer, raw_response, question_id, prompt_id, complete):
        meta = self.sample_meta[question_id]
        thinking = self._extract_thinking_content(raw_response)
        thinking_citation_count = len(self._extract_citations(thinking))
        response_citation_count = len(self._extract_citations(answer))
        resolved_thinking_citation_count = self._count_resolved_citations(thinking, question_id)
        resolved_response_citation_count = self._count_resolved_citations(answer, question_id)
        references = {
            "thinking": self._build_reference_entries(thinking, question_id),
            "response": self._build_reference_entries(answer, question_id),
        }
        boxed_answer = extract_last_boxed_content(answer)
        try:
            score = compute_score(
                data_source=meta["data_source"],
                solution_str=answer,
                ground_truth=meta["ground_truth"],
                extra_info=meta["extra_info"],
            )
        except Exception:
            score = 0.0
        question = self.questions[question_id]
        if hasattr(question, "tolist"):
            question = question.tolist()
        context_lengths = self.context_cnt[question_id]
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0.0
        information_lengths = self.information_cnt[question_id]
        information_length_total = sum(information_lengths)
        information_length = information_length_total / len(information_lengths) if information_lengths else 0.0
        answer_length = tokenize(answer)
        answer_file = f"{self.answer_dir}/result_{meta['file_stub']}.json"
        with open(answer_file, 'w', encoding='utf-8') as f:
            result = {
                    "model": self.model_name,
                    "data_source": meta["data_source"],
                    "id": meta["sample_id"],
                    "question": question,
                    "prompt_id": prompt_id,
                    "complete": complete,
                    "thinking": thinking,
                    "answer": answer,
                    "reference": references,
                    "ground_truth": meta["ground_truth"],
                    "pred_boxed_answer": boxed_answer,
                    "score": score,
                    "turns": self.turn_id[question_id],
                    "search count": self.search_cnt[question_id],
                    "summary count": self.summary_cnt[question_id],
                    "context lengths": self.context_cnt[question_id],
                    "context_length": avg_context_length,
                    "information_length": information_length,
                    "information_length_total": information_length_total,
                    "answer_length": answer_length,
                    "extra_info": meta["extra_info"],
                    "thinking citation count": thinking_citation_count,
                    "response citation count": response_citation_count,
                    "resolved thinking citation count": resolved_thinking_citation_count,
                    "resolved response citation count": resolved_response_citation_count,
                }
            json.dump(result, f, indent=4)

    def _search(self, query, num_docs, search_log, question_id):
        
        if self.search_engine == 'clueweb':
            documents = query_clueweb(query, num_docs=num_docs)
        elif self.search_engine == 'serper':
            documents = query_serper(query, num_docs=num_docs)
        elif self.search_engine == 'medcorp':
            documents = query_medcorp(query, num_docs=num_docs)
        elif self.search_engine == 'fineweb':
            documents = query_fineweb(query, num_docs=num_docs)
        else:
            raise ValueError(f"Invalid search engine: {self.search_engine}")
        info_retrieved, indexed_docs = self._format_retrieved_documents(
            documents, self.turn_id[question_id]
        )

        if self.verbose:
            with open(search_log, 'a', encoding='utf-8') as f:
                f.write(f"[turn={self.turn_id[question_id]}]\n")
                f.write(f"query:\n{query}\n\n")
                f.write(f"info_retrieved:\n{info_retrieved}\n\n\n")
        return info_retrieved, indexed_docs

def sanitize_for_filename(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))

def extract_last_boxed_content(text):
    matches = re.findall(r"\\boxed\{(.*?)\}", text or "", flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()

def load_samples_from_medmix_parquet(file_path, sample_limit=500, sample_seed=42, data_sources=None):
    df = pd.read_parquet(file_path)
    if data_sources:
        requested_sources = {str(source) for source in data_sources}
        df = df[df["data_source"].astype(str).isin(requested_sources)]
    if sample_limit is not None:
        # sample_limit may be an int (same cap for every source) or a dict
        # mapping data_source -> int (per-source caps).
        sampled_parts = []
        for source_name, source_df in df.groupby("data_source", dropna=False, sort=False):
            cap = (
                sample_limit.get(str(source_name))
                if isinstance(sample_limit, dict)
                else sample_limit
            )
            if cap is not None and cap > 0 and len(source_df) > cap:
                sampled_parts.append(source_df.sample(n=cap, random_state=sample_seed))
            else:
                sampled_parts.append(source_df)
        if sampled_parts:
            df = pd.concat(sampled_parts, axis=0, ignore_index=True)
    df = df.reset_index(drop=True)

    samples = []
    for row_idx, row in df.iterrows():
        data_source = row.get("data_source", "unknown")
        extra_info = row.get("env_kwargs", {}).get("extra_info", {})
        question = extra_info.get("question")

        reward_model = row.get("reward_model", {})
        prompt = row.get("prompt")

        if not isinstance(extra_info, dict):
            extra_info = {}
        if not isinstance(reward_model, dict):
            reward_model = {}

        sample_id = extra_info["qid"]
        question = extra_info["question"]
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
                "task_description": question,
                "ground_truth": reward_model.get("ground_truth"),
                "extra_info": extra_info,
                "template_with_info": CALC_TEMPLATE_WITH_INFO if use_calc_template else QA_TEMPLATE_WITH_INFO,
                "template_no_info": CALC_TEMPLATE_NO_INFO if use_calc_template else QA_TEMPLATE_NO_INFO,
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

def build_stats_summary(records, data_source):
    n = len(records)
    if n == 0:
        return None

    search_total = sum(float(r.get("search count", 0.0)) for r in records)
    info_total = sum(float(r.get("information_length_total", 0.0)) for r in records)

    def mean(field):
        return sum(float(r.get(field, 0.0)) for r in records) / n

    def nonzero_proportion(field):
        return sum(int(r.get(field, 0) or 0) > 0 for r in records) / n

    return {
        "data_source": data_source,
        "num_samples": n,
        "score": mean("score"),
        "context_length": mean("context_length"),
        "search_count": mean("search count"),
        "search_proportion": nonzero_proportion("search count"),
        "summary_count": mean("summary count"),
        "summary_proportion": nonzero_proportion("summary count"),
        "turns": mean("turns"),
        "answer_length": mean("answer_length"),
        "information_length": info_total / search_total if search_total > 0 else 0.0,
        "thinking_citation_nonzero_proportion": nonzero_proportion("thinking citation count"),
        "response_citation_nonzero_proportion": nonzero_proportion("response citation count"),
        "resolved_thinking_citation_nonzero_proportion": nonzero_proportion("resolved thinking citation count"),
        "resolved_response_citation_nonzero_proportion": nonzero_proportion("resolved response citation count"),
        "mean_thinking_citation_count": mean("thinking citation count"),
        "mean_response_citation_count": mean("response citation count"),
        "mean_resolved_thinking_citation_count": mean("resolved thinking citation count"),
        "mean_resolved_response_citation_count": mean("resolved response citation count"),
    }

def write_stats_result_jsonl(answer_dir, result_jsonl_path):
    result_files = sorted(glob.glob(os.path.join(answer_dir, "result_*.json")))
    grouped = {}
    for file_path in tqdm(result_files, desc="Scoring results", unit="file"):
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
            summary = build_stats_summary(records, data_source)
            if summary is not None:
                f.write(json.dumps(summary, ensure_ascii=True) + "\n")

        if all_records:
            overall = build_stats_summary(all_records, "__overall__")
            if overall is not None:
                f.write(json.dumps(overall, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/path/to/data/medmix/test.parquet', help='Path to MedMix test.parquet file')
    parser.add_argument('--log_dir', type=str, default='/path/to/evaluation/verl_agent_w_rubrics/logs_medcorp', help='Log directory')
    parser.add_argument('--answer_dir', type=str, default='/path/to/evaluation/verl_agent_w_rubrics/results_medcorp', help='Result directory')
    parser.add_argument('--model', default='qwen_local', help='base model: gemini | qwen | gpt | bedrock | qwen_local')
    parser.add_argument('--model_path', type=str, default='/path/to/SFT/qwen3-1.7b/output_6964878/checkpoint-2661')
    parser.add_argument('--url', type=str, default='http://localhost:8000/v1', help='URL to use')
    parser.add_argument('--search_engine', type=str, default='medcorp', help='Search engine to use: clueweb | serper | medcorp | fineweb')
    parser.add_argument('--max_turns', type=int, default=15, help='Max number of turns')
    parser.add_argument('--num_docs', type=int, default=6, help='Number of documents/results to retrieve per search')
    parser.add_argument('--result_jsonl', type=str, default=None, help='Path to write aggregated stats jsonl. Defaults to {answer_dir}/result.jsonl')
    parser.add_argument('--sample_limit', type=int, default=100, help='Maximum number of samples per data_source from test.parquet')
    parser.add_argument('--data_source', nargs='+', default=None, help='Only process specified data_source values. Supports multiple values, e.g. --data_source source_a source_b')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    answer_dir = args.answer_dir
    log_dir = args.log_dir
    model = args.model
    url = args.url
    search_engine = args.search_engine
    # make sure answer_dir and log_dir exist
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    result_jsonl_path = args.result_jsonl or os.path.join(answer_dir, "result.jsonl")

    # Load samples from MedMix parquet
    samples = load_samples_from_medmix_parquet(
        args.data_path,
        sample_limit=args.sample_limit,
        sample_seed=42,
        data_sources=args.data_source,
    )
    total_questions = len(samples)
    print(f"Loaded {total_questions} samples from {args.data_path}")

    # Filter out completed questions
    remaining_samples, completed_count = filter_completed_samples(samples, answer_dir)
    remaining_questions_num = len(remaining_samples)
    
    print(f"Total dataset: {total_questions} questions")
    print(f"Already completed: {completed_count} questions")
    print(f"Remaining to process: {remaining_questions_num} questions")
    
    # If no questions to process, exit
    if remaining_questions_num == 0:
        write_stats_result_jsonl(answer_dir, result_jsonl_path)
        print(f"Aggregated stats jsonl written to {result_jsonl_path}")
        print("All questions have been completed!")
        exit(0)

    max_turns = args.max_turns
    config = {
              "max_turns": max_turns, # Max number of turns
              "num_docs": args.num_docs, # Number of documents to retrieve
            } 
    
    agent = LLMAgent(
        config,
        log_dir=log_dir,
        answer_dir=answer_dir,
        verbose=True,
        model=model,
        search_engine=search_engine,
        url=url,
        result_jsonl=result_jsonl_path,
        model_path=args.model_path,
    )

    agent.run_llm_loop_parallel(remaining_samples)
    write_stats_result_jsonl(answer_dir, result_jsonl_path)
    print(f"Aggregated stats jsonl written to {result_jsonl_path}")
