# FaithMed: Training LLMs For Faithful Evidence-Based Medical Reasoning

### 1. Environment Setup

#### 1.1 Create Conda Environment

```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install vllm==0.11.0

pip3 install flash-attn --no-build-isolation --no-cache-dir
cd FaithMed/verl-agent
pip install -e .

# SFT training 
pip install llamafactory==0.9.4
```

#### 1.2 Set API Keys

* Save your API keys to `keys.env`:

```bash
echo 'GEMINI_API_KEY="your_gemini_api_key"' >> ~/keys.env
echo 'WANDB_API_KEY="your_wandb_api_key"' >> ~/keys.env
```

* Export `KEYS_ENV_PATH` (required by training scripts):

```bash
# Add to ~/.bashrc for persistence:
echo 'export KEYS_ENV_PATH=~/keys.env' >> ~/.bashrc
source ~/.bashrc
# Or export temporarily for the current session only:
export KEYS_ENV_PATH=/path/to/your/keys.env
```

### 2. Data Preparation

Prepare MedMix Data for RL: including MedQA, MedMCQA, MedCalcBench, HeadQA, MMLU-Pro-Health, MedBullets, MedXpertQA

```bash
python data_preparation/medmix_preparation.py \
    --save_dir /path/to/output \        # (required) directory to save output parquet files
    --output_name medmix \              # folder name created under save_dir (default: medmix)
    --train_limit 10000 10000 10000 10000 \  # cap training set per source: MedQA HeadQA MedMCQA MedCalc (default: 10000 each)
    --test_limit \                      # flag: cap test set to 200 rows per data_source
    --remove_sft \                      # flag: exclude questions already used in SFT training
    --sft_answers_dir /path/to/sft      # dir of accepted SFT answer JSON files (used with --remove_sft)
```

### 3. SFT Data Generation and Training

#### 3.1 Generate SFT Trajectories

Generate agentic search trajectories used as SFT supervision signal:

```bash
python -m search_agent.sft.generate_trajectories \
    --model bedrock \
    --search_engine medcorp \
    --sample_limit 1000 500 1000 500 \
    --workers 8
```

Or via Slurm:

```bash
sbatch medrm_scripts/sft_generate_trajectories.sh
```

#### 3.2 SFT Training

Update the paths in `search_agent/sft/train_sft.yaml`:

```yaml
model_name_or_path: /path/to/Qwen3-1.7B
dataset_dir: /path/to/SFT       # folder containing dataset_info.json and sft_data.jsonl
output_dir: /path/to/SFT/output
```

Then run training with LLaMA-Factory:

```bash
NUM_GPUS=4 bash search_agent/sft/run_sft.sh
```

Or via Slurm:

```bash
sbatch medrm_scripts/sft_train.sh
```

### 4. Agentic RL Pipeline

#### 4.1 RL Training

Start RL Training

```bash
bash verl-agent/examples/faithmed_trainer/run_medrm.sh 
    --validation_process_reward_freq 60 \
    --verbose_freq 10 \
    --process_reward_enable 1 \
    --step_scoring True # enable step-level process reward
```

Or via Slurm:

```bash
sbatch medrm_scripts/faithmed_rl_sbatch.sh
```

#### 4.2 Acknowledgements

We thank the developers of [`verl`](https://github.com/volcengine/verl) and [`verl-agent`](https://github.com/langfengQ/verl-agent) for their open-source RL training infrastructure.
