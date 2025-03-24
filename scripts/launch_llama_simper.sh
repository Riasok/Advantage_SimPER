#!/bin/bash

# ========== HYPERPARAMETERS ==========
# Model configuration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BETA=0.1
LEARNING_RATE=1e-6
USE_PEFT=true

# Training configuration
BATCH_SIZE=16
GRAD_ACCUM_STEPS=16
EVAL_BATCH_SIZE=16
N_EPOCHS=1
EVAL_EVERY=1000
ENABLE_INTERMEDIATE_CHECKPOINTS=true

# Hardware configuration
GPU_DEVICES="1,2"  # Default to "1,2" if not provided

# Dataset configuration
DATASETS="[data/dpomath.json]"
CACHE_DIR="~/reasoning/Advantage_SimPER/outputs"
TEST_DATASET="math500"
NUM_SAMPLES_PER_PROMPT=8 # for sampling

# Output naming
EXP_NAME="llama3.1-8b-simper-test-${BETA}-${LEARNING_RATE}"
OUTPUT_FILE="outputs/llama3.1-8b-simper-${BETA}-${LEARNING_RATE}.json"

# ========== ENVIRONMENT SETUP ==========
export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
export MODEL_PATH=${MODEL_NAME}
export CKPT=${CACHE_DIR}/${EXP_NAME}/FINAL
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

GPU_COUNT=$(echo ${GPU_DEVICES} | tr -cd ',' | wc -c)
GPU_COUNT=$((GPU_COUNT + 1))  # Count is commas + 1
CONFIG_FILE="accelerate_config/fsdp_${GPU_COUNT}gpu.yaml"
echo "Using ${GPU_COUNT} GPUs with config: ${CONFIG_FILE}"

# ========== TRAINING ==========
accelerate launch \
  --config_file ${CONFIG_FILE} \
  launch.py \
  loss=simper \
  model=llama exp_name=${EXP_NAME} \
  datasets=${DATASETS} \
  ++cache_dir=${CACHE_DIR} \
  ++model.name_or_path=${MODEL_PATH} \
  ++lr=${LEARNING_RATE} \
  ++loss.beta=${BETA} \
  ++model.batch_size=${BATCH_SIZE} \
  ++model.gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
  ++model.eval_batch_size=${EVAL_BATCH_SIZE} \
  ++config.intermediate_checkpoints=${ENABLE_INTERMEDIATE_CHECKPOINTS} \
  ++config.eval_every=${EVAL_EVERY} \
  ++model.use_peft=${USE_PEFT} \
  ++n_epochs=${N_EPOCHS}

# ========== EVALUATION ==========
echo "Starting evaluation on ${TEST_DATASET}"
python -m train.sample ${CKPT} \
  --gpu_count ${GPU_COUNT} \
  --output_file ${OUTPUT_FILE} \
  --datasets ${TEST_DATASET} \
  --num_samples_per_prompt ${NUM_SAMPLES_PER_PROMPT} \
  --split test

echo "Training and evaluation complete"