#! /bin/bash

USER_ENV=`whoami`
set -x
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MEDRM_DIR=$(cd "$VERL_AGENT_DIR/.." && pwd)
cd "$VERL_AGENT_DIR"
export PYTHONPATH=$MEDRM_DIR:${PYTHONPATH}
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1


export PROJECT_NAME=verl_agent
export WANDB_API_KEY=$(grep -E "^WANDB_API_KEY=" "${KEYS_ENV_PATH}" | cut -d'=' -f2- | tr -d '"'"'"' ')
export WANDB_OFFICIAL=1
# export WANDB_MODE=offline 
export ARNOLD_WORKER_NUM=1 # number of nodes you want to use 

# FaithMed grouped RL config
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
enable_similarity=True # enable similarity-based step grouping
SIMILARITY_THRESH=0.8 # similarity threshold for step grouping

# Default values
RUN_NAME=faithmed_rl
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=10000
MAX_RESPONSE_LENGTH=3072
LEARNING_RATE=1e-6
PPO_MINI_BATCH_SIZE=128
N_GPUS_PER_NODE=4
# per GPU
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=128
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=1
DATASET_NAME=medmix_wo_hard  
ROLLOUT_GPU_MEMORY_UTIL=0.8
LOG_VAL_GENERATIONS=3
SAVE_FREQ=20
TEST_FREQ=20
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
ENV_MAX_STEPS=4
NUM_CPUS_PER_ENV_WORKER=0.125
SEARCH_ENGINE=medcorp
NUM_DOCS=6
PROCESS_REWARD_ENABLE=0     # 1=on, 0=off
PROCESS_REWARD_LAMBDA=0.05  # λ weight for process reward
PROCESS_REWARD_DEBUG=0      # print first N trajectories' grading details (0=silent)
GIGPO_ANCHOR_SOURCE=query  # retrieved_doc or query
GEMINI_MODEL=gemini-2.5-flash-lite
USE_INVALID_ACTION_PENALTY=True
INVALID_ACTION_PENALTY_COEF=0.05
VERBOSE_FREQ=-1
VALIDATION_PROCESS_REWARD_FREQ=-1
STEP_SCORING=False
RUBRIC_DISABLED_DIMS=""
GAMMA=1.0
PPO_MAX_TOKEN_LEN_PER_GPU=32000
# file dir
HDFS_DATA_PATH="/path/to/data"
HDFS_CHECKPOINT_PATH="/path/to/checkpoints"
HDFS_LOG_PATH="/path/to/logs"
MODEL_PATH="/path/to/sft_checkpoint"

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --validation_process_reward_freq) VALIDATION_PROCESS_REWARD_FREQ="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --hdfs_data_path) HDFS_DATA_PATH="$2"; shift 2 ;;
    --hdfs_ckpt_path) HDFS_CHECKPOINT_PATH="$2"; shift 2 ;;
    --hdfs_log_path) HDFS_LOG_PATH="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --log_val_generations) LOG_VAL_GENERATIONS="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2 ;;
    --env_max_steps) ENV_MAX_STEPS="$2"; shift 2 ;;
    --num_cpus_per_env_worker) NUM_CPUS_PER_ENV_WORKER="$2"; shift 2 ;;
    --search_engine) SEARCH_ENGINE="$2"; shift 2 ;;
    --num_docs) NUM_DOCS="$2"; shift 2 ;;
    --process_reward_enable) PROCESS_REWARD_ENABLE="$2"; shift 2 ;;
    --process_reward_lambda) PROCESS_REWARD_LAMBDA="$2"; shift 2 ;;
    --process_reward_debug) PROCESS_REWARD_DEBUG="$2"; shift 2 ;;
    --gigpo_anchor_source) GIGPO_ANCHOR_SOURCE="$2"; shift 2 ;;
    --gemini_model) GEMINI_MODEL="$2"; shift 2 ;;
    --use_invalid_action_penalty) USE_INVALID_ACTION_PENALTY="$2"; shift 2 ;;
    --invalid_action_penalty_coef) INVALID_ACTION_PENALTY_COEF="$2"; shift 2 ;;
    --verbose_freq) VERBOSE_FREQ="$2"; shift 2 ;;
    --step_scoring) STEP_SCORING="$2"; shift 2 ;;
    --rubric_disabled_dims) RUBRIC_DISABLED_DIMS="$2"; shift 2 ;;
    --similarity_thresh) SIMILARITY_THRESH="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --ppo_max_token_len_per_gpu) PPO_MAX_TOKEN_LEN_PER_GPU="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done


# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"
ENV_LOG_DIR="$HDFS_LOG_PATH/$RUN_NAME/logs_train"
ENV_ANSWER_DIR="$HDFS_LOG_PATH/$RUN_NAME/results_train"
ENV_GROUPS_DIR="$HDFS_LOG_PATH/$RUN_NAME/groups_train"
ENV_RUBRICS_DIR="$HDFS_LOG_PATH/$RUN_NAME/rubrics_train"
ENV_VAL_LOG_DIR="$HDFS_LOG_PATH/$RUN_NAME/logs_valid"
ENV_VAL_ANSWER_DIR="$HDFS_LOG_PATH/$RUN_NAME/results_valid"
ENV_VAL_RUBRICS_DIR="$HDFS_LOG_PATH/$RUN_NAME/rubrics_valid"

echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE" 
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "Rollout N: $ROLLOUT_N" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Path: $MODEL_PATH"
echo "Remove Clip: $REMOVE_CLIP"
echo "LOG FILE PATH: $LOG_FILE_PATH"
echo "LOG VAL GENERATIONS: $LOG_VAL_GENERATIONS"
echo "Validation Process Reward Freq: $VALIDATION_PROCESS_REWARD_FREQ"
echo "Final RUN_NAME: $RUN_NAME"
echo "Final LOG_FILE_PATH: $LOG_FILE_PATH"
echo "Process Reward Enable: $PROCESS_REWARD_ENABLE"
echo "Process Reward Lambda: $PROCESS_REWARD_LAMBDA"
echo "Process Reward Debug:  $PROCESS_REWARD_DEBUG"
echo "Step Group Anchor Source: $GIGPO_ANCHOR_SOURCE"
echo "Gemini Model: $GEMINI_MODEL"
echo "Use Invalid Action Penalty: $USE_INVALID_ACTION_PENALTY"
echo "Invalid Action Penalty Coef: $INVALID_ACTION_PENALTY_COEF"
echo "Verbose Freq: $VERBOSE_FREQ"
echo "Step Scoring: $STEP_SCORING"
echo "Rubric Disabled Dims: $RUBRIC_DISABLED_DIMS"
echo "Similarity Thresh: $SIMILARITY_THRESH"
echo "Gamma: $GAMMA"
echo "PPO Max Token Len Per GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: $VAL_BATCH_SIZE\nMax Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Epochs: $TOTAL_EPOCHS\nDataset Name: $DATASET_NAME\nModel Path: $MODEL_PATH"

unset ROCR_VISIBLE_DEVICES

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.compute_mean_std_cross_all_data=True \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.actor.use_invalid_action_penalty=$USE_INVALID_ACTION_PENALTY \
  actor_rollout_ref.actor.invalid_action_penalty_coef=$INVALID_ACTION_PENALTY_COEF \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True  \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=$GAMMA \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=$mode \
  algorithm.gigpo.enable_similarity=$enable_similarity \
  algorithm.gigpo.similarity_thresh=$SIMILARITY_THRESH \
  env.env_name=medrm \
  env.seed=0 \
  env.rollout.n=$ROLLOUT_N \
  env.search_engine=$SEARCH_ENGINE \
  env.num_docs=$NUM_DOCS \
  env.max_steps=$ENV_MAX_STEPS \
  env.process_reward_enable=$PROCESS_REWARD_ENABLE \
  env.process_reward_lambda=$PROCESS_REWARD_LAMBDA \
  env.process_reward_debug=$PROCESS_REWARD_DEBUG \
  env.gigpo_anchor_source=$GIGPO_ANCHOR_SOURCE \
  env.gemini_model=$GEMINI_MODEL \
  env.verbose_freq=$VERBOSE_FREQ \
  env.step_scoring=$STEP_SCORING \
  env.rubric_disabled_dims=$RUBRIC_DISABLED_DIMS \
  env.log_dir=$ENV_LOG_DIR \
  env.answer_dir=$ENV_ANSWER_DIR \
  env.groups_dir=$ENV_GROUPS_DIR \
  env.rubrics_dir=$ENV_RUBRICS_DIR \
  env.val_log_dir=$ENV_VAL_LOG_DIR \
  env.val_answer_dir=$ENV_VAL_ANSWER_DIR \
  env.val_rubrics_dir=$ENV_VAL_RUBRICS_DIR \
  env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.log_val_generations=$LOG_VAL_GENERATIONS \
  trainer.validation_process_reward_freq=$VALIDATION_PROCESS_REWARD_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.max_actor_ckpt_to_keep=5 \
  trainer.max_critic_ckpt_to_keep=5 \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.val_before_train=True 2>&1 | tee -a $LOG_FILE_PATH
