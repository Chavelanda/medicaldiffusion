#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Reconstruct images
# python -m evaluation.reconstruct_images dataset=allcts model=vq_gan_3d dataset.save_path=data/test

# Build meshes
python -m evaluation.build_meshes dataset=allcts model=vq_gan_3d dataset.root_dir=data/test +dataset.spacing="[0.3345859, 0.46262616, 0.49944812]"

# Calculate FID
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.root_dir=data/test

# Calculate MS-SSIM
# python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim


