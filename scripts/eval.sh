
#!/bin/bash

# Parameter for desirable weight
WEIGHTD=$1


# Setup environment variables
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export CKPT=~/.cache/huggingface/hub/llama3-dpo-01-D-1e-5/FINAL
#export CKPT=~/.cache/huggingface/hub/meta-llama/Llama-3.1-8B-Instruct

export HF_DATASETS_OFFLINE=1    
export HF_HUB_OFFLINE=1


python -m train.sample $CKPT \
  --gpu_count 1 \
  --output_file outputs/llama3-8b-dpo-0.1-${WEIGHTD}D-1e-5.json \
  --datasets math500 \
  --num_samples_per_prompt 8 \
  --split test