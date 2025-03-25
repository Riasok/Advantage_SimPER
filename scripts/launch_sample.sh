
#!/bin/bash

# Setup environment variables
export CUDA_VISIBLE_DEVICES="0,1"
export HYDRA_FULL_ERROR=1
#export MODEL_NAME=llama3-dpo-test-D-1e-5
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export CKPT=/outputs/${MODEL_NAME}/FINAL
#export CKPT=~/.cache/huggingface/hub/meta-llama/Llama-3.1-8B-Instruct

export HF_DATASETS_OFFLINE=1    
export HF_HUB_OFFLINE=1


python -m train.sample $MODEL_NAME \
  --gpu_count 1 \
  --output_file data/llama_math.json \
  --datasets math500 \
  --num_samples_per_prompt 8 \
  --split train

python label.py data/llama_math.json binary
python label.py data/llama_math.json pairwise