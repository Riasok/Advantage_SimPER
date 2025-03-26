
#!/bin/bash

# Setup environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HYDRA_FULL_ERROR=1
#export MODEL_NAME=llama3-dpo-test-D-1e-5
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export CKPT=/outputs/${MODEL_NAME}/FINAL
#export CKPT=~/.cache/huggingface/hub/meta-llama/Llama-3.1-8B-Instruct

export HF_DATASETS_OFFLINE=0   
export HF_HUB_OFFLINE=0


python -m train.sample ${MODEL_NAME} \
  --gpu_count 4 \
  --output_file data/llama_math.json \
  --datasets hendrycks_math \
  --num_samples_per_prompt 8 \
  --split train

python label_.py data/llama_math.json binary
# python label_.py data/llama_math.json pairwise