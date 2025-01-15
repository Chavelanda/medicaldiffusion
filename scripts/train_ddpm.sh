#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CPATH="/usr/local/cuda-11.1/include:$CPATH"
export CUDA_VISIBLE_DEVICES=0

# Activate Python virtual environment
source .venv/bin/activate

# Run Python script
# python -m train.train_ddpm dataset=allcts model=ddpm
python -m train.train_ddpm +experiment=ddpm-t1000