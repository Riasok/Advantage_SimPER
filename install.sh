#!/bin/bash

# conda create --name halos python=3.10.14 -y
# conda activate halos
conda install pip -y
pip install packaging ninja
ninja --version

# Detect CUDA version
if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
  MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1)
  MINOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f2)
  echo "Detected CUDA version: $CUDA_VERSION"
else
  echo "CUDA not found. Please install CUDA and try again."
  exit 1
fi

# Set environment variables based on detected CUDA version
if [[ "$MAJOR_VERSION" == "12" && "$MINOR_VERSION" == "6" ]]; then
  echo "Setting up for CUDA 12.6..."
  export CUDA_HOME=/usr/local/cuda-12.6
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  TORCH_VERSION="torch==2.4.0+cu126"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
elif [[ "$MAJOR_VERSION" == "12" && "$MINOR_VERSION" == "4" ]]; then
  echo "Setting up for CUDA 12.4..."
  export CUDA_HOME=/usr/local/cuda-12.4
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  TORCH_VERSION="torch==2.4.0+cu121"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
fi

echo "Installing PyTorch: $TORCH_VERSION"
pip install $TORCH_VERSION --index-url $TORCH_INDEX_URL

# Flash-Attn installation
echo "Installing flash-attn..."
pip install flash-attn==2.6.3 --no-build-isolation --force-reinstall

# Other libraries
echo "Installing additional libraries..."
pip install transformers==4.46.2
pip install peft==0.12.0
pip install datasets==2.20.0
pip install accelerate==0.33.0
pip install vllm==0.6.3.post1
pip install alpaca-eval immutabledict langdetect wandb omegaconf openai hydra-core==1.3.2

# Skipping lm-eval installation as requested
echo "Setup complete!"

# Uncomment if you want to load the datasets without lm-eval
# python << EOF
# from datasets import load_dataset
# load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
# EOF