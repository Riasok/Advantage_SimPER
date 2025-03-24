
#!/bin/bash

# Parameter for desirable weight
WEIGHTD=$1


# Setup environment variables
export CUDA_VISIBLE_DEVICES=1,2
export HYDRA_FULL_ERROR=1
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export CKPT=/data1/home/minjaeoh/reasoning/HALOs/outputs/llama3-8b-simper-test-${WEIGHTD}D-3e-5/FINAL
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Activate your conda environment if needed
# conda activate your_environment

# Run the training (using only one GPU)
accelerate launch \
  --config_file accelerate_config/fsdp_2gpu.yaml \
  launch.py \
  loss=simper \
  model=llama exp_name=llama3-8b-simper-test-${WEIGHTD}D-3e-5 \
  datasets=[examples/dpomath.json] \
  ++cache_dir=/data1/home/minjaeoh/reasoning/HALOs/outputs \
  ++model.name_or_path=$MODEL_PATH \
  ++lr=1e-6 \
  ++loss.beta=0.1 \
  ++model.batch_size=16 ++model.gradient_accumulation_steps=16 ++model.eval_batch_size=16 \
  ++loss.desirable_weight=${WEIGHTD} \
  ++config.intermediate_checkpoints=true \
  ++config.eval_every=1000 \
  ++model.use_peft=true \
  ++n_epochs=1

python -m train.sample $CKPT \
  --gpu_count 2 \
  --output_file outputs/llama3-8b-simper-0.1-${WEIGHTD}D-3e-6.json \
  --datasets math500 \
  --num_samples_per_prompt 8 \
  --split test

  #++cache_dir=/data1/home/minjaeoh/.cache/huggingface/hub \