
#!/bin/bash

# Parameter for desirable weight
WEIGHTD=$1


# Setup environment variables
# export CUDA_VISIBLE_DEVICES=1,3
export HYDRA_FULL_ERROR=1
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export CKPT=./models/qwen2-5-3B-instruct-kto-01-${WEIGHTD}D-5e-6/FINAL
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

# Activate your conda environment if needed
# conda activate your_environment

# Run the training (using only one GPU)
accelerate launch \
  --config_file accelerate_config/fsdp_4gpu.yaml \
  launch.py \
  loss=dpo \
  model=llama exp_name=llama3-dpo-01-${WEIGHTD}D-5e-6 \
  datasets=[examples/pairwise_feedback.json] \
  # ++cache_dir=/home/minjaeoh/.cache/huggingface/hub \
  ++model.name_or_path=$MODEL_PATH \
  ++lr=5e-6 \
  ++loss.beta=0.1 \
  ++model.batch_size=8 ++model.gradient_accumulation_steps=4 ++model.eval_batch_size=8 \
  ++loss.desirable_weight=${WEIGHTD} \
  ++config.intermediate_checkpoints=true \
  ++config.eval_every=500 \
  ++model.use_peft=true
