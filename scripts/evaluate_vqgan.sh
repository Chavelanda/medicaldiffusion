#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=1
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Reconstruct images. Uses training dataset config
# python -m evaluation.reconstruct_images model=reconstruct dataset=allcts dataset.split=test +dataset.save_path="data/allcts-051-512-up-conv-ae"
python -m evaluation.reconstruct_images model=reconstruct dataset=allcts +dataset.save_path="data/allcts-051-512-up-only-recon-ae" dataset.split=test

# Build meshes. Uses training dataset config
# python -m evaluation.build_meshes dataset=allcts dataset.root_dir=data/allcts-051-216-ae dataset.resample=1 +dataset.spacing="[1.2138, 1.2138, 1.2138]" model=reconstruct

# Calculate FID between test dataset config and train dataset config
# python -m evaluation.calculate_fid model=fid model.run_name="fid-allcts-vqgan-07-train-med3d-01" model.extractor="med3d" model.batch_size=50 dataset=allcts 

# Calculate MS-SSIM
# python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim dataset.root_dir=data/allcts-lf-07


