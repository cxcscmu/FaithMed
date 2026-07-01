# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import torch
import os
import re
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


"""
FaithMed adaptation of GiGPO-style grouped policy optimization.

This module keeps the upstream GiGPO grouping structure while adding
FaithMed-specific outcome/process reward handling, similarity-based grouping,
and group logging utilities. The original upstream-style implementation is kept
in core_gigpo.py for attribution and compatibility.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute returns for each step. (Eq. 5 in the paper)

    Formula: R_t = process_reward_t + outcome_reward_t + γ * Σ_{s>t} γ^(s-t-1) * outcome_reward_s

    Each step's return is its own process reward plus the discounted sum of all future
    outcome rewards (episode-level rewards). Future process rewards are NOT accumulated —
    only the current step's process reward contributes.

    Since outcome reward is typically non-zero only at the final step of a trajectory,
    this reduces to:  R_t = process_reward_t + γ^(T-1-t) * outcome_reward_{T-1}

    Args:
        batch (DataProto): Input batch. Must contain 'process_reward' and 'outcome_reward'
            in non_tensor_batch.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Per-step returns.
    """
    process_rewards = batch.non_tensor_batch['process_reward'].astype(np.float32)
    outcome_rewards = batch.non_tensor_batch['outcome_reward'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        traj_indices = np.where(traj_uids == uid)[0]

        traj_process = process_rewards[traj_indices]
        traj_outcome = outcome_rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"

        traj_returns = np.zeros_like(traj_process)
        # Backward pass: accumulate only outcome rewards (not process rewards)
        outcome_running = 0.0
        for t in reversed(range(len(traj_process))):
            traj_returns[t] = traj_process[t] + traj_outcome[t] + gamma * outcome_running
            outcome_running = traj_outcome[t] + gamma * outcome_running

        returns_by_traj[uid] = traj_returns

    all_returns = np.zeros_like(process_rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]
        all_returns[i] = returns_by_traj[uid][idx_in_traj]

    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)

    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores, step_group_uids, episode_advantages, step_advantages


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return step_advantages


def log_gigpo_groups(
    batch: DataProto,
    tokenizer,
    groups_dir: str,
    train_step: int,
    invalid_action_penalty_coef: float = 0.1,
    lambda_weight: float = 0.05,
    step_scoring: bool = False,
):
    """
    Write one markdown file per step_group_uid under groups_dir/step_{train_step}/.
    File names: {safe_ds}_{safe_qid}_g{group_idx}.md.
    Each step is identified as {safe_ds}_{safe_qid}_{rollout_idx}_t{turn_n},
    matching the file_stub convention of logs/results/rubrics.
    Per-rubric process-reward breakdown is included for each step.
    """
    if step_scoring:
        from rewards.step_rubric_judge import load_step_rubrics, step_score_contributions
        rubrics = load_step_rubrics()
    else:
        from rewards.rubrics_judge import load_active_rubrics, step_rubric_contributions
        rubrics = load_active_rubrics()

    step_dir = os.path.join(groups_dir, f"step_{train_step}")
    os.makedirs(step_dir, exist_ok=True)

    step_group_uids      = batch.non_tensor_batch['step_group_uid']
    ep_adv_scalar        = batch.non_tensor_batch['_gigpo_episode_adv']
    step_adv_scalar      = batch.non_tensor_batch['_gigpo_step_adv']
    data_sources         = batch.non_tensor_batch.get('data_source')
    question_ids         = batch.non_tensor_batch.get('question_id')
    episode_uids         = batch.non_tensor_batch.get('uid')
    is_action_valid      = batch.non_tensor_batch.get('is_action_valid')
    step_types           = batch.non_tensor_batch.get('step_type')
    anchor_obs_arr       = batch.non_tensor_batch.get('anchor_obs')
    anchor_retrieved_arr = batch.non_tensor_batch.get('anchor_retrieved_obs')
    anchor_query_arr     = batch.non_tensor_batch.get('anchor_query')
    outcome_rewards      = batch.non_tensor_batch.get('outcome_reward')
    process_rewards      = batch.non_tensor_batch.get('process_reward')
    traj_uids            = batch.non_tensor_batch.get('traj_uid')
    episode_rewards      = batch.non_tensor_batch.get('episode_rewards')
    raw_rewards          = batch.non_tensor_batch.get('rewards')
    rubric_raw_scores_arr = batch.non_tensor_batch.get('rubric_raw_scores')

    step_rewards_tensor = batch.batch['step_rewards'] if 'step_rewards' in batch.batch.keys() else None

    # Single pass: derive rollout_idx and turn_n for every step.
    # traj_info[t_uid] = [rollout_idx, turn_counter]
    traj_info: Dict[str, list] = {}
    ep_traj_count: Dict[str, int] = defaultdict(int)
    rollout_idxs: List[int] = []
    turn_ns: List[int] = []
    for i in range(len(step_group_uids)):
        ep_uid = str(episode_uids[i]) if episode_uids is not None else "unknown"
        t_uid  = str(traj_uids[i])    if traj_uids  is not None else str(i)
        if t_uid not in traj_info:
            traj_info[t_uid] = [ep_traj_count[ep_uid], 0]
            ep_traj_count[ep_uid] += 1
        rollout_idxs.append(traj_info[t_uid][0])
        turn_ns.append(traj_info[t_uid][1])
        traj_info[t_uid][1] += 1

    # group step indices by step_group_uid
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, uid in enumerate(step_group_uids):
        groups[uid].append(i)

    # assign sequential group_idx within each episode_uid
    ep_uid_to_group_uids: Dict[str, List[str]] = defaultdict(list)
    for group_uid, indices in groups.items():
        ep_uid = str(episode_uids[indices[0]]) if episode_uids is not None else "unknown"
        ep_uid_to_group_uids[ep_uid].append(group_uid)
    group_uid_to_idx: Dict[str, int] = {
        g_uid: idx
        for group_list in ep_uid_to_group_uids.values()
        for idx, g_uid in enumerate(group_list)
    }

    for group_uid, indices in groups.items():
        ds  = str(data_sources[indices[0]]) if data_sources is not None else "unknown"
        qid = str(question_ids[indices[0]]) if question_ids is not None else "unknown"
        safe_ds  = re.sub(r"[^0-9A-Za-z._-]+", "_", ds)
        safe_qid = re.sub(r"[^0-9A-Za-z._-]+", "_", qid)
        g_idx    = group_uid_to_idx.get(group_uid, 0)
        filepath = os.path.join(step_dir, f"{safe_ds}_{safe_qid}_g{g_idx}.md")

        rep_obs = str(anchor_obs_arr[indices[0]]) if anchor_obs_arr is not None else ""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# GiGPO Step Group\n\n")
            f.write(f"**Question ID:** {qid}  \n")
            f.write(f"**Group Index:** g{g_idx}  \n")
            f.write(f"**Train Step:** {train_step}  \n")
            f.write(f"**Group Size:** {len(indices)} steps\n\n")

            for rank, i in enumerate(indices, 1):
                f.write(f"---\n\n## Step {rank}/{len(indices)}\n\n")

                traj_str  = f"{safe_ds}_{safe_qid}_{rollout_idxs[i]}_t{turn_ns[i]}"
                step_type = str(step_types[i]) if step_types is not None else "unknown"

                ep_adv   = float(ep_adv_scalar[i])
                st_adv   = float(step_adv_scalar[i])
                ep_rew   = float(episode_rewards[i]) if episode_rewards is not None else 0.0
                raw_rew  = float(raw_rewards[i]) if raw_rewards is not None else 0.0
                disc_rew = float(step_rewards_tensor[i]) if step_rewards_tensor is not None else 0.0
                out_r    = float(outcome_rewards[i]) if outcome_rewards is not None else 0.0
                proc_r   = float(process_rewards[i]) if process_rewards is not None else 0.0
                valid    = bool(is_action_valid[i]) if is_action_valid is not None else True
                inv_pen  = 0.0 if valid else invalid_action_penalty_coef

                this_obs  = str(anchor_obs_arr[i]) if anchor_obs_arr is not None else ""
                this_retrieved = str(anchor_retrieved_arr[i]) if anchor_retrieved_arr is not None else ""
                this_query = str(anchor_query_arr[i]) if anchor_query_arr is not None else ""
                anchor_type = (
                    "query" if this_query and this_obs == this_query
                    else "retrieved_doc" if this_retrieved and this_obs == this_retrieved
                    else "state"
                )
                sim_score = SequenceMatcher(None, rep_obs, this_obs).ratio() if rep_obs else 1.0

                f.write("| Field | Value |\n|-------|-------|\n")
                f.write(f"| Trajectory | `{traj_str}` |\n")
                f.write(f"| Step Type | `{step_type}` |\n")
                f.write(f"| Anchor Type | `{anchor_type}` |\n")
                f.write(f"| Episode Advantage | {ep_adv:.4f} |\n")
                f.write(f"| Step Advantage | {st_adv:.4f} |\n")
                f.write(f"| Similarity to Group Rep | {sim_score:.4f} |\n\n")

                f.write("### Query\n\n```\n")
                f.write(this_query)
                f.write("\n```\n\n")

                f.write("### Retrieved Observation\n\n```\n")
                f.write(this_retrieved)
                f.write("\n```\n\n")

                f.write("### Rewards\n\n")
                f.write("| Metric | Value |\n|--------|-------|\n")
                f.write(f"| Episode Reward (total) | {ep_rew:.4f} |\n")
                f.write(f"| Step Reward (raw) | {raw_rew:.4f} |\n")
                f.write(f"| Step Reward (discounted return) | {disc_rew:.4f} |\n")
                f.write(f"| Outcome Reward | {out_r:.4f} |\n")
                f.write(f"| Process Reward | {proc_r:.4f} |\n")
                f.write(f"| Invalid Penalty | {inv_pen:.4f} |\n\n")

                # per-rubric process-reward breakdown
                raw_scores: dict = {}
                if rubric_raw_scores_arr is not None and rubric_raw_scores_arr[i]:
                    try:
                        raw_scores = json.loads(rubric_raw_scores_arr[i])
                    except Exception:
                        pass
                if raw_scores and rubrics:
                    f.write("### Rubric Breakdown\n\n")
                    if step_scoring:
                        rubric_rows = step_score_contributions(raw_scores, rubrics, lambda_weight)
                        f.write("| ID | Rubric | Dim | Raw | Contribution |\n")
                        f.write("|----|--------|-----|-----|--------------|\n")
                        for row in rubric_rows:
                            f.write(f"| {row['id']} | {row['name']} | {row['dimension']} "
                                    f"| {row['raw']} | {row['contrib']:.4f} |\n")
                    else:
                        rubric_rows = step_rubric_contributions(raw_scores, step_type, rubrics, lambda_weight)
                        f.write("| ID | Rubric | Raw | Applicable | Contribution |\n")
                        f.write("|----|--------|-----|------------|--------------|\n")
                        for row in rubric_rows:
                            f.write(f"| {row['id']} | {row['name']} | {row['raw']} "
                                    f"| {'yes' if row['applicable'] else 'no'} "
                                    f"| {row['contrib']:.4f} |\n")
                    f.write("\n")

                response_text = tokenizer.decode(batch.batch['responses'][i], skip_special_tokens=True)

                f.write("### Response\n\n```\n")
                f.write(response_text)
                f.write("\n```\n\n")
