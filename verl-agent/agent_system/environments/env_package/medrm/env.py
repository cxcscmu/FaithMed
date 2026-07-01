import concurrent.futures
import json
import os
import re
import threading
from datetime import datetime

from search_agent.retrieval import query_clueweb, query_serper, query_medcorp
from rewards.medmix_reward import compute_score
from rewards.rubrics_judge import load_active_rubrics, score_trajectory, configure_debug, configure_gemini_model
from rewards.step_rubric_judge import (
    score_step,
    load_step_rubrics,
    extract_step_features,
    filter_step_rubrics_by_disabled_dims,
)

# Limit concurrent medcorp requests to avoid overwhelming the server.
# 128 simultaneous requests causes server-side queuing (last request waits 10s+).
# With max_concurrent=20, throughput stays high but tail latency drops significantly.
_SEARCH_SEMAPHORE = threading.Semaphore(64)

# Shared executor for background rubric scoring (Step 1 of process reward).
# Trajectories submit their Gemini call here as soon as done=True, so scoring
# overlaps with the ongoing rollout of other trajectories.
_RUBRIC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32)
from agent_system.environments.prompts.medrm import (
    QA_TEMPLATE_NO_INFO,
    QA_TEMPLATE_WITH_INFO,
    CALC_TEMPLATE_NO_INFO,
    CALC_TEMPLATE_WITH_INFO,
    FORMAT_REMINDER_PROMPT,
)

MAX_CONTEXT_LENGTH = 8000


class MedRMEnv:
    """
    Single-trajectory environment.
    One instance handles exactly one question at a time.
    reset() starts a new question; step() advances it one turn.
    """

    def __init__(self, config):
        self.config = config
        self.search_engine = config.get("search_engine", "medcorp")
        self.num_docs = config.get("num_docs", 6)
        self.step_scoring = bool(config.get("step_scoring", False))
        configure_debug(int(config.get("process_reward_debug", 0)))
        configure_gemini_model(config.get("gemini_model", "gemini-2.5-flash-lite"))
        self.verbose_freq = int(config.get("verbose_freq", -1))  # verbose_freq: log when train_step % verbose_freq == 0; -1 = never
        self.log_dir = config.get("log_dir") or ""
        self.answer_dir = config.get("answer_dir") or ""
        self.rubrics_dir = config.get("rubrics_dir") or ""
        self._train_step = 0
        self._verbose = False  # set per-episode in reset() based on _train_step
        if self.verbose_freq > 0:
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            if self.answer_dir:
                os.makedirs(self.answer_dir, exist_ok=True)
            if self.rubrics_dir:
                os.makedirs(self.rubrics_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, question, question_id, rollout_idx, ground_truth,
              data_source, extra_info):
        """
        Start a new episode for one question.

        Args:
            question:       the full prompt string (from parquet's 'prompt' column)
            question_id:    unique id for this sample
            rollout_idx:    which rollout copy this is (for GRPO)
            ground_truth:   ground truth expected by compute_score
            data_source:    e.g. "medqa", "ncbi/MedCalc-Bench"
            extra_info:     dict with 'qid', 'question' (task description), etc.

        Returns:
            state (str), info (dict)
        """
        self.question = question          # full prompt (what the model sees first)
        self.question_id = question_id
        self.rollout_idx = rollout_idx
        self.ground_truth = ground_truth
        self.data_source = data_source
        self.extra_info = extra_info or {}

        # task_description is the bare question text (used for template formatting)
        self.task_description = self.extra_info.get("question", question)

        # Choose QA vs CALC template based on data source
        is_calc = str(data_source) == "ncbi/MedCalc-Bench"
        self._template_no_info = CALC_TEMPLATE_NO_INFO if is_calc else QA_TEMPLATE_NO_INFO
        self._template_with_info = CALC_TEMPLATE_WITH_INFO if is_calc else QA_TEMPLATE_WITH_INFO

        # Build initial state (no history yet)
        self.state = self._template_no_info.format(
            task_description=self.task_description,
            memory_context="",
        )

        # Episode state
        self.turn_id = 0
        self.search_cnt = 0
        self.consecutive_search_cnt = 0
        self.memory_context = []        # list of "[Turn N]:\n..." strings
        self.need_format_reminder = False
        self.done = False
        self.rewards = []
        self._rubric_future: concurrent.futures.Future | None = None
        # GiGPO: per-step type labels for process-reward distribution.
        # One entry appended per active env step; see _determine_step_type().
        self.step_types: list[str] = []
        self._prev_retrieved_passages: list[str] = []   # search results from completed turns
        self._step_rubric_futures: list = []            # Future|None per active step (step_scoring mode)
        self._step_features: list = []                  # features dict|None per active step (for logging)
        self._prev_env_feedback: str = ""
        self._prev_search_query: str = ""
        self._verbose = (
            self.verbose_freq > 0 and self._train_step % self.verbose_freq == 0
        )
        if self._verbose:
            if self.log_dir:
                os.makedirs(os.path.join(self.log_dir, f"step_{self._train_step}"), exist_ok=True)
            if self.answer_dir:
                os.makedirs(os.path.join(self.answer_dir, f"step_{self._train_step}"), exist_ok=True)

        self.info = {
            "question_id": question_id,
            "data_source": data_source,
            "ground_truth": ground_truth,
            "search_cnt": 0,
            "consecutive_search_cnt": 0,
            "won": False,
            "environment_feedback": "",
            "step_type": "reset",
        }
        return self.state, self.info

    def set_train_step(self, train_step: int):
        self._train_step = train_step

    def step(self, original_response, action):
        """
        Advance the episode one turn.

        Args:
            original_response: the raw model output (including <think> if any)
            action:            parsed action string from projection.py,
                               e.g. "<search>...</search>" or "<answer>...</answer>",
                               or None if the format was invalid.

        Returns:
            next_input (str), reward (float), done (bool), info (dict)
        """
        if self.done:
            return "", 0.0, True, self.info

        self.turn_id += 1
        prev_passages_snapshot = list(self._prev_retrieved_passages)

        # Execute the action → get observation
        prev_search_cnt = self.search_cnt
        anchor_retrieved_obs = self.state if self.turn_id == 1 else self._prev_env_feedback
        anchor_query = "" if self.turn_id == 1 else self._prev_search_query
        done, next_obs, action_type, action_content = self._execute_response(action)
        next_anchor_query = action_content if action_type == "search" else ""
        self.info["environment_feedback"] = next_obs
        self.info["tool_calling"] = self.search_cnt > prev_search_cnt

        # GiGPO: label this step by the anchor that entered it (prev env feedback).
        # turn 1 → anchor = initial prompt ("reset"); turn ≥ 2 → anchor = _prev_env_feedback.
        if self.turn_id == 1:
            step_type = "reset"
        elif self._prev_env_feedback == "Invalid action, cannot be executed.":
            step_type = "invalid"
        elif done:
            step_type = "answer"
        else:
            step_type = "search"
        self.step_types.append(step_type)
        self._prev_env_feedback = next_obs
        self._prev_search_query = next_anchor_query
        self.info["step_type"] = step_type
        self.info["anchor_retrieved_obs"] = anchor_retrieved_obs
        self.info["anchor_query"] = anchor_query
        self.info["next_anchor_query"] = next_anchor_query

        # Per-step scoring uses prev_passages_snapshot (snapped before _execute_response,
        # so current turn's search result is excluded per GiGPO step-scoring semantics).
        if self.step_scoring:
            self._submit_step_scoring(original_response, action, step_type, prev_passages_snapshot)
        # Append current search result so the NEXT step's snapshot includes it.
        # Include reset steps: turn 1 always gets step_type="reset" even when the
        # model issues a <search>, so its result must also be captured here.
        m = re.search(r"<information>(.*?)</information>", next_obs, re.DOTALL)
        if m:
            self._prev_retrieved_passages.append(m.group(1).strip())

        # Record trajectory before state is updated
        if self._verbose and self.log_dir:
            self._record_trajectory(self.state, original_response, next_obs)

        # Build next prompt
        next_input = self._update_input(original_response, next_obs)

        # Reward: only at terminal step when answer was given
        if done:
            answer = self._extract_answer(original_response)
            reward = compute_score(
                data_source=self.data_source,
                solution_str=answer,
                ground_truth=self.ground_truth,
                extra_info=self.extra_info,
            )
            self.done = True
            self.info["won"] = reward > 0.0
            if not self.step_scoring:
                self._submit_rubric_scoring()
        elif self.turn_id >= self.config.get("max_turns", 6):
            # Hit turn limit without answer → terminal, reward = 0
            reward = 0.0
            self.done = True
            if not self.step_scoring:
                self._submit_rubric_scoring()
        else:
            reward = 0.0

        self.rewards.append(reward)
        self.info["search_cnt"] = self.search_cnt
        self.info["consecutive_search_cnt"] = self.consecutive_search_cnt

        if self.done and self._verbose and self.answer_dir:
            final_answer = self._extract_answer(original_response)
            self._log_result(final_answer, reward)

        return next_input, reward, self.done, self.info

    def _execute_response(self, action):
        """
        Execute the parsed action.

        Returns:
            done (bool), next_obs (str), action_type (str | None), action_content (str)
        """
        if action is None:
            self.need_format_reminder = True
            return False, "Invalid action, cannot be executed.", None, ""

        action_type, content = self._parse_action(action)

        if action_type == "answer":
            return True, "Answer generated, the process is done.", action_type, content

        elif action_type == "search":
            self.search_cnt += 1
            self.consecutive_search_cnt += 1
            search_results = self._search(content)
            return False, f"<information>{search_results}</information>", action_type, content

        else:
            # Should not happen given projection.py filtering
            self.need_format_reminder = True
            return False, "Invalid action, cannot be executed.", action_type, content

    def _parse_action(self, action):
        """Extract (action_type, content) from '<type>content</type>'."""
        # Strip leading <think> block
        action = re.sub(r"^\s*<think>.*?</think>\s*", "", action, flags=re.DOTALL)
        start_open = action.find('<')
        start_close = action.find('>', start_open)
        action_type = action[start_open + 1:start_close]
        end_open = action.rfind('</')
        content = action[start_close + 1:end_open].strip()
        return action_type, content

    def _update_input(self, cur_response, next_obs):
        """
        Build the next model input by appending this turn to the history.
        """
        if self.need_format_reminder:
            context = f"[Turn {self.turn_id}]:\n{cur_response}\n\n"
            context += FORMAT_REMINDER_PROMPT
            new_input = self.state + context
            self.need_format_reminder = False
        else:
            context = f"[Turn {self.turn_id}]:\n{cur_response}\n{next_obs}\n"
            self.memory_context.append(context)
            memory_str = "\n".join(self.memory_context).strip()

            template = (
                self._template_with_info if self.search_cnt > 0
                else self._template_no_info
            )
            new_input = template.format(
                task_description=self.task_description,
                memory_context=memory_str,
            )

        self.state = new_input
        return new_input

    def _extract_answer(self, response):
        """Return the content inside <answer>...</answer>, or the full response."""
        m = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def _submit_rubric_scoring(self):
        """
        Submit Step 1 of process reward (Gemini rubric scoring) to the shared
        background executor immediately when a trajectory terminates.

        The Future is stored in self._rubric_future and collected later by
        MedRMMultiProcessEnv.compute_process_rewards() once the full batch is done.

        Skipped entirely when process_reward_enable=0 to avoid unnecessary API costs.
        """
        if not int(self.config.get("process_reward_enable", 1)):
            self._rubric_future = None
            return
        rubrics = load_active_rubrics()
        memory_context = list(self.memory_context)  # snapshot before any reset
        self._rubric_future = _RUBRIC_EXECUTOR.submit(
            score_trajectory, memory_context, rubrics
        )

    def _submit_step_scoring(self, response: str, action: str | None, step_type: str, prev_passages: list[str]):
        """Submit per-step rubric scoring to the background executor (step_scoring mode)."""
        if not int(self.config.get("process_reward_enable", 1)) or step_type == "invalid":
            self._step_rubric_futures.append(None)
            self._step_features.append(None)
            return
        full_rubrics = load_step_rubrics()
        scoring_rubrics = filter_step_rubrics_by_disabled_dims(
            full_rubrics, self.config.get("rubric_disabled_dims", "")
        )
        features = extract_step_features(response, prev_passages)
        self._step_features.append(features)
        if not scoring_rubrics:
            self._step_rubric_futures.append(None)
            return
        self._step_rubric_futures.append(
            _RUBRIC_EXECUTOR.submit(
                score_step, response, action, prev_passages, step_type, scoring_rubrics, features
            )
        )

    def get_memory_context(self) -> list[str]:
        """Return accumulated turn strings for rubric scoring."""
        return list(self.memory_context)

    def _search(self, query):
        """Call the configured search engine."""
        try:
            with _SEARCH_SEMAPHORE:
                if self.search_engine == "clueweb":
                    documents = query_clueweb(query, num_docs=self.num_docs)
                elif self.search_engine == "serper":
                    documents = query_serper(query, num_docs=self.num_docs)
                elif self.search_engine == "medcorp":
                    documents = query_medcorp(query, num_docs=self.num_docs)
                else:
                    raise ValueError(f"Unknown search engine: {self.search_engine}")
        except Exception as e:
            return f"Search error: {e}"

        blocks = []
        for idx, doc in enumerate(documents, start=1):
            blocks.append(f"Result [T{self.turn_id}-R{idx}]:\n{(doc or '').strip()}")
        info_retrieved = "\n\n".join(blocks)

        if self._verbose and self.log_dir:
            file_stub = self._file_stub
            search_log = os.path.join(self.log_dir, f"step_{self._train_step}", f"search_{file_stub}.log")
            with open(search_log, 'a', encoding='utf-8') as f:
                f.write(f"[turn={self.turn_id}]\n")
                f.write(f"query:\n{query}\n\n")
                f.write(f"info_retrieved:\n{info_retrieved}\n\n\n")

        return info_retrieved

    def _record_trajectory(self, input_text, response, next_obs):
        file_stub = self._file_stub
        step_dir = os.path.join(self.log_dir, f"step_{self._train_step}")
        trajectory_log = os.path.join(step_dir, f"trajectory_{file_stub}.md")
        trajectory_jsonl_log = os.path.join(step_dir, f"trajectory_{file_stub}.jsonl")

        input_length = len(input_text.split())
        response_length = len(response.split())

        if not os.path.exists(trajectory_log) or os.path.getsize(trajectory_log) == 0:
            with open(trajectory_log, 'w', encoding='utf-8') as f:
                f.write(f"## Question ID: {self.question_id} (Rollout {self.rollout_idx})\n\n")

        with open(trajectory_log, 'a', encoding='utf-8') as f:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"## Turn {self.turn_id} {time_str}\n\n")
            f.write(f"### Input:\n**length={input_length}**\n{input_text}\n\n")
            f.write(f"### Response:\n**length={response_length}**\n{response}\n\n--------------------------------\n\n")

        with open(trajectory_jsonl_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "turn": self.turn_id,
                "input": input_text,
                "response": response,
                "next_obs": next_obs,
                "context_length": input_length + response_length,
            }) + '\n')

    def _log_result(self, answer, reward):
        file_stub = self._file_stub
        answer_file = os.path.join(self.answer_dir, f"step_{self._train_step}", f"result_{file_stub}.json")
        with open(answer_file, 'w', encoding='utf-8') as f:
            json.dump({
                "data_source": self.data_source,
                "question_id": self.question_id,
                "rollout_idx": self.rollout_idx,
                "answer": answer,
                "ground_truth": self.ground_truth,
                "reward": reward,
                "rewards": self.rewards,
                "won": self.info.get("won", False),
                "turns": self.turn_id,
                "search_cnt": self.search_cnt,
            }, f, indent=4)

    @property
    def _file_stub(self):
        safe_ds = re.sub(r"[^0-9A-Za-z._-]+", "_", str(self.data_source))
        safe_qid = re.sub(r"[^0-9A-Za-z._-]+", "_", str(self.question_id))
        return f"{safe_ds}_{safe_qid}_{self.rollout_idx}"

    def _rubric_score_entries(self, raw_scores: dict, rubrics: list, extra_fields: tuple = ()) -> dict:
        """Build the rubric_scores sub-dict shared by both log methods."""
        return {
            r["id"]: {
                "name":  r.get("name", r["id"]),
                "type":  r.get("type", "binary"),
                **{k: r.get(k) for k in extra_fields},
                "score": raw_scores.get(r["id"], -1),
                "na":    raw_scores.get(r["id"], -1) == -1,
            }
            for r in rubrics
        }

    def log_rubrics(self, raw_scores: dict, rubrics: list):
        """Write trajectory-level rubric scores + Gemini prompt to rubrics_dir when verbose."""
        if not self._verbose or not self.rubrics_dir:
            return
        from rewards.rubrics_judge import extract_features, _build_prompt
        step_dir = os.path.join(self.rubrics_dir, f"step_{self._train_step}")
        os.makedirs(step_dir, exist_ok=True)
        features   = extract_features(self.memory_context)
        applicable = [r for r in rubrics if raw_scores.get(r["id"], -1) != -1]
        prompt     = _build_prompt(applicable, features) if applicable else ""
        with open(os.path.join(step_dir, f"rubric_{self._file_stub}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "question_id":   self.question_id,
                "data_source":   self.data_source,
                "rollout_idx":   self.rollout_idx,
                "won":           self.info.get("won", False),
                "rubric_scores": self._rubric_score_entries(raw_scores, rubrics),
                "prompt":        prompt,
            }, f, indent=4)

    def log_step_rubrics(self, step_raw_scores: list[dict], rubrics: list):
        """Write per-step rubric scores + Gemini prompt to rubrics_dir when verbose.

        One file per active step: rubric_{file_stub}_turn{N}.json
        """
        if not self._verbose or not self.rubrics_dir:
            return
        from rewards.rubrics_judge import _build_prompt
        step_dir = os.path.join(self.rubrics_dir, f"step_{self._train_step}")
        os.makedirs(step_dir, exist_ok=True)
        for turn_id, (raw_scores, features) in enumerate(
            zip(step_raw_scores, self._step_features), start=1
        ):
            if features is None:   # invalid step — no Gemini call was made
                continue
            applicable = [r for r in rubrics if raw_scores.get(r["id"], -1) != -1]
            prompt     = _build_prompt(applicable, features) if applicable else ""
            fname = os.path.join(step_dir, f"rubric_{self._file_stub}_turn{turn_id}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({
                    "question_id":   self.question_id,
                    "data_source":   self.data_source,
                    "rollout_idx":   self.rollout_idx,
                    "turn_id":       turn_id,
                    "won":           self.info.get("won", False),
                    "rubric_scores": self._rubric_score_entries(raw_scores, rubrics, extra_fields=("dimension",)),
                    "prompt":        prompt,
                }, f, indent=4)
