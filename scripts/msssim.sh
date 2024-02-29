#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=2
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Run Python script
# python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim

python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim dataset.metadata_name="metadata2.csv"
python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim dataset.metadata_name="metadata3.csv"
python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim dataset.metadata_name="metadata4.csv"
python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim dataset.metadata_name="metadata5.csv"
python -m evaluation.calculate_msssim dataset=allcts-msssim model=msssim dataset.metadata_name="metadata6.csv"