import asyncio
import concurrent.futures
import sys

import gymnasium as gym

from agent_system.environments.env_package.medrm.env import MedRMEnv


class MedRMWorker:
    """
    One worker = one MedRMEnv instance = one trajectory at a time.
    """
    def __init__(self, config):
        self.env = MedRMEnv(config)

    def reset(self, question, question_id, rollout_idx, ground_truth,
              data_source, extra_info):
        return self.env.reset(question, question_id, rollout_idx,
                              ground_truth, data_source, extra_info)

    def step(self, original_response, action):
        return self.env.step(original_response, action)

    def set_train_step(self, train_step: int):
        self.env.set_train_step(train_step)


class MedRMMultiProcessEnv(gym.Env):
    """
    Manages env_num * group_n workers in parallel.

    - env_num:  number of distinct questions per training step
    - group_n:  rollout copies per question (for GRPO)
    - Total parallel trajectories = env_num * group_n
    """

    def __init__(self, seed, env_num, group_n, config, resources_per_worker=None):
        super().__init__()
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.config = config

        print(
            f"Init MedRMMultiProcessEnv: env_num={env_num}, "
            f"group_n={group_n}, total_workers={self.num_processes}",
            file=sys.stderr,
        )
        self.workers = [MedRMWorker(config) for _ in range(self.num_processes)]
        max_workers = min(self.num_processes, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def reset(self, questions, question_ids, ground_truths,
              data_sources, extra_infos):
        """
        Reset all workers in parallel.

        Args:
            questions:     list of length <= num_processes  (full prompt strings)
            question_ids:  list of length <= num_processes
            ground_truths: list of length <= num_processes
            data_sources:  list of length <= num_processes
            extra_infos:   list of length <= num_processes

        Each worker receives its own entry directly (1-to-1 mapping).
        The caller is expected to pass env_num * group_n items, with the
        same question repeated group_n times consecutively (interleaved),
        matching the layout produced by DataProto.repeat(interleave=True).
        rollout_idx is derived as i % group_n for each worker index i.
        If fewer than num_processes items are provided (e.g. the last batch),
        dummy entries are padded and filtered out before returning.
        """
        actual_n = len(questions)
        if actual_n > self.num_processes:
            raise ValueError(
                f"Got {actual_n} questions but num_processes={self.num_processes}"
            )

        # Pad to num_processes with dummy entries if needed
        pad_n = self.num_processes - actual_n
        if pad_n > 0:
            questions     = list(questions)     + [""] * pad_n
            question_ids  = list(question_ids)  + [""] * pad_n
            ground_truths = list(ground_truths) + [""] * pad_n
            data_sources  = list(data_sources)  + ["unknown"] * pad_n
            extra_infos   = list(extra_infos)   + [{}] * pad_n

        tasks = []
        for i, worker in enumerate(self.workers):
            rollout_idx = i % self.group_n
            tasks.append(
                self._loop.run_in_executor(
                    self._executor,
                    worker.reset,
                    questions[i],
                    question_ids[i],
                    rollout_idx,
                    ground_truths[i],
                    data_sources[i],
                    extra_infos[i],
                )
            )

        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]

        # Mark dummy workers as already done so step() short-circuits instantly
        # (no search calls, no generation cost) for the padded slots.
        if pad_n > 0:
            for i in range(actual_n, self.num_processes):
                self.workers[i].env.done = True
            obs_list  = obs_list[:actual_n]
            info_list = info_list[:actual_n]

        self._actual_n = actual_n
        print(f"Reset {actual_n} MedRM workers (padded {pad_n})", file=sys.stderr)
        return obs_list, info_list

    def step(self, original_responses, actions):
        """
        Step all workers in parallel.

        Args:
            original_responses: list[str] of length <= num_processes
            actions:            list[str|None] of length <= num_processes
                                (parsed by medrm_projection)
        """
        actual_workers = len(original_responses)
        if actual_workers > self.num_processes:
            raise ValueError(
                f"Got {actual_workers} responses but num_processes={self.num_processes}"
            )

        # Pad to num_processes with empty strings for dummy workers.
        # Dummy workers have done=True so step() returns instantly (no cost).
        pad_n = self.num_processes - actual_workers
        if pad_n > 0:
            original_responses = list(original_responses) + [""] * pad_n
            actions = list(actions) + [None] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, worker.step, resp, act)
            for worker, resp, act in zip(self.workers, original_responses, actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        # Drop dummy worker results
        if pad_n > 0:
            obs_list    = obs_list[:actual_workers]
            reward_list = reward_list[:actual_workers]
            done_list   = done_list[:actual_workers]
            info_list   = info_list[:actual_workers]

        return obs_list, reward_list, done_list, info_list

    def compute_process_rewards(self, lambda_weight: float = 0.05, return_raw_scores: bool = False):
        """
        Compute λ-weighted process rewards for all trajectories in the current batch.

        Step 1 (slow Gemini calls) was already submitted to _RUBRIC_EXECUTOR in the
        background by each MedRMEnv the moment its trajectory terminated.  This method
        only needs to:
          1. Collect those futures (blocks only for calls still in-flight).
          2. Run the fast per-group mean-centering (Step 2).

        Workers are laid out as:
            workers[q * group_n + r]  →  question q, rollout r
        so workers[q*group_n : (q+1)*group_n] form one GRPO group.

        Returns a flat list of length actual_n.
        """
        from rewards.rubrics_judge import load_active_rubrics, apply_group_centering

        actual_n = getattr(self, "_actual_n", self.num_processes)
        rubrics  = load_active_rubrics()

        # ── Step 1: collect background rubric-scoring futures ─────────────
        # debug_strs are printed here (main thread) so Ray forwards them to console.
        raw_scores: list[dict] = []
        for i in range(actual_n):
            env    = self.workers[i].env
            future = getattr(env, "_rubric_future", None)
            if future is not None:
                try:
                    scores, debug_str = future.result()
                    raw_scores.append(scores)
                    if debug_str is not None:
                        print(debug_str, flush=True)
                except Exception as exc:
                    print(f"[MedRMMultiProcessEnv] rubric scoring failed for "
                          f"worker {i}: {exc}")
                    scores = {r["id"]: -1 for r in rubrics}
                    raw_scores.append(scores)
            else:
                scores = {r["id"]: -1 for r in rubrics}
                raw_scores.append(scores)
            env.log_rubrics(scores, rubrics)

        # ── Step 2: per-group mean-centering (fast) ───────────────────────
        process_rewards: list[float] = [0.0] * actual_n
        q = 0
        while q * self.group_n < actual_n:
            start = q * self.group_n
            end   = min(start + self.group_n, actual_n)
            process_rewards[start:end] = apply_group_centering(
                raw_scores[start:end], rubrics, lambda_weight
            )
            q += 1

        if return_raw_scores:
            return process_rewards, raw_scores
        return process_rewards

    def compute_step_process_rewards(self, lambda_weight: float = 0.05) -> tuple[list[list[float]], list]:
        """
        Compute per-step process rewards for GiGPO (no group centering).

        Two modes controlled by config["step_scoring"]:
          False (default): score the full trajectory once, distribute to steps.
          True:            score each step immediately after generation; collect here.

        Returns:
            (step_rewards, raw_scores_list):
                step_rewards:    list[list[float]] — outer=trajectory, inner=per-active-step
                raw_scores_list: list[dict]        — representative raw scores per trajectory
        """
        from rewards.rubrics_judge import load_active_rubrics, aggregate_trajectory_rubric_score

        actual_n     = getattr(self, "_actual_n", self.num_processes)
        step_scoring = bool(self.config.get("step_scoring", False))

        if step_scoring:
            from rewards.step_rubric_judge import (
                load_step_rubrics,
                aggregate_step_scores,
                filter_step_rubrics_by_disabled_dims,
            )
            rubrics = load_step_rubrics()
            scoring_rubrics = filter_step_rubrics_by_disabled_dims(
                rubrics, self.config.get("rubric_disabled_dims", "")
            )
        else:
            rubrics = load_active_rubrics()
            scoring_rubrics = rubrics

        fallback = {r["id"]: -1 for r in rubrics}

        def _resolve(future, label=""):
            if future is None:
                return dict(fallback)
            try:
                scores, debug_str = future.result()
                if debug_str:
                    print(debug_str, flush=True)
                expanded_scores = dict(fallback)
                expanded_scores.update(scores)
                return expanded_scores
            except Exception as exc:
                print(f"[MedRMMultiProcessEnv] {label}: {exc}")
                return dict(fallback)

        result: list[list[float]] = []
        raw_scores_list: list[dict] = []

        for i in range(actual_n):
            env = self.workers[i].env

            if step_scoring:
                step_scores_list = []
                last_scores      = dict(fallback)
                for future in env._step_rubric_futures:
                    scores = _resolve(future, f"step scoring worker {i}")
                    last_scores = scores
                    step_scores_list.append(scores)
                step_rewards = [aggregate_step_scores(s, scoring_rubrics, lambda_weight) for s in step_scores_list]
                env.log_step_rubrics(step_scores_list, rubrics)
            else:
                scores       = _resolve(getattr(env, "_rubric_future", None), f"worker {i}")
                last_scores  = scores
                traj_score   = aggregate_trajectory_rubric_score(scores, rubrics, lambda_weight)
                step_rewards = [traj_score] * len(env.step_types)
                env.log_rubrics(last_scores, rubrics)

            raw_scores_list.append(step_scores_list if step_scoring else last_scores)
            result.append(step_rewards)

        return result, raw_scores_list

    def set_train_step(self, train_step: int):
        """Broadcast the current training step to all workers so they can gate verbose logging."""
        for worker in self.workers:
            worker.set_train_step(train_step)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._executor.shutdown(wait=True)
        self._loop.close()
        self.workers.clear()
        self._closed = True


def build_medrm_envs(seed=0, env_num=1, group_n=1, config=None, resources_per_worker=None):
    env_config = dict(config) if config else {}
    return MedRMMultiProcessEnv(seed, env_num, group_n, env_config, resources_per_worker)
