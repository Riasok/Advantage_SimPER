#!/bin/bash

# conda create --name halos python=3.10.14 -y
# conda activate halos

conda install pip -y
pip install packaging ninja
ninja --version

# CUDA 경로 설정 (시스템 환경에 따라 경로 확인 필요)
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# PyTorch 2.4 (CUDA 12.4과 호환되도록 system CUDA 사용)
pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Flash-Attn 설치 (CUDA 12.4 대응)
pip install flash-attn==2.6.3 --no-build-isolation --force-reinstall

# 기타 라이브러리
pip install transformers==4.46.2
pip install peft==0.12.0
pip install datasets==2.20.0
pip install accelerate==0.33.0
pip install vllm==0.6.3.post1
pip install alpaca-eval immutabledict langdetect wandb omegaconf openai hydra-core==1.3.2

# lm-eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
python << EOF
from lm_eval import tasks
task_names = ["winogrande", "mmlu", "gsm8k_cot", "bbh_cot_fewshot", "arc_easy", "arc_challenge", "hellaswag", "ifeval"]
task_dict = tasks.get_task_dict(task_names)
from datasets import load_dataset
load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
EOF
