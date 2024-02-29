#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=2
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Run Python script
# python -m generate.generate_filtered model=generate dataset=allcts

python -m generate.generate_filtered model=generate dataset=allcts model.class_idx=0 model.n_samples=167 
python -m generate.generate_filtered model=generate dataset=allcts model.class_idx=1 model.n_samples=212
python -m generate.generate_filtered model=generate dataset=allcts model.class_idx=2 model.n_samples=154
python -m generate.generate_filtered model=generate dataset=allcts model.class_idx=3 model.n_samples=156
python -m generate.generate_filtered model=generate dataset=allcts model.class_idx=4 model.n_samples=38