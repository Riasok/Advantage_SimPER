
#!/bin/bash

# Setup environment variables
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export MODEL_NAME=llama3-dpo-test-D-1e-5
export CKPT=/outputs/${MODEL_NAME}/FINAL
#export CKPT=~/.cache/huggingface/hub/meta-llama/Llama-3.1-8B-Instruct

export HF_DATASETS_OFFLINE=1    
export HF_HUB_OFFLINE=1


python -m train.sample $CKPT \
  --gpu_count 1 \
  --output_file outputs/${MODEL_NAME}.json \
  --datasets math500 \
  --num_samples_per_prompt 8 \
  --split test