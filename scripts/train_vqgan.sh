#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0,1
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Run Python script
# python -m train.train_vqgan dataset=allcts model=vq_gan_3d

python -m train.train_vqvae_upsampling +experiment=ae-up-res-conv
