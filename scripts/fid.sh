#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Run Python script
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.qs=[2] model.stats_dir="evaluation/fid/stats/allcts-qs2/"
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.qs=[3] model.stats_dir="evaluation/fid/stats/allcts-qs3/"
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.qs=[4] model.stats_dir="evaluation/fid/stats/allcts-qs4/"
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.qs=[5] model.stats_dir="evaluation/fid/stats/allcts-qs5/"
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.qs=[6] model.stats_dir="evaluation/fid/stats/allcts-qs6/"
# python -m evaluation.calculate_fid_stats model=fid dataset=allcts dataset.qs=[2,3,4,5] model.stats_dir="evaluation/fid/stats/allcts-qs2-5/"

# python -m evaluation.calculate_fid model=fid dataset=allcts model.run_name="fid-allcts-vqgan-07-train-vqvae-01" model.extractor="vqvae" model.batch_size=50
# python -m evaluation.calculate_fid model=fid dataset=allcts model.run_name="fid-allcts-vqgan-07-train-med3d-01" model.extractor="med3d" model.batch_size=50
# python -m evaluation.calculate_fid model=fid dataset=allcts model.run_name="fid-allcts-vqgan-07-train-stunet-01" model.extractor="stunet" model.batch_size=20
# python -m evaluation.calculate_fid model=fid dataset=allcts model.run_name="fid-allcts-vqgan-07-train-misfm-01" model.extractor="misfm" model.batch_size=15

python -m evaluation.calculate_fid model=fid dataset=allcts model.run_name="fid-allcts-051-216-gen-med3d" model.extractor="med3d" model.batch_size=30