#!/bin/bash

# Set CUDA environment variables
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=1
export CPATH="/usr/local/cuda-11.1/include:$CPATH"

# Activate Python virtual environment
source .venv/bin/activate

# Run Python script
# python -m generate.generate_filtered model=generate dataset=allcts

# python -m generate.generate_filtered model=generate dataset=mrnet model.m=5 model.class_idx=[] model.name_prefix=mrnet-lf-01-000 model.n_samples=217 model.train_latents=generate/latents/mrnet_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=mrnet model.m=5 model.class_idx=0 model.name_prefix=mrnet-lf-01-100 model.n_samples=433 model.train_latents=generate/latents/mrnet_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=mrnet model.m=5 model.class_idx=[0,2] model.name_prefix=mrnet-lf-01-101 model.n_samples=272 model.train_latents=generate/latents/mrnet_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=mrnet model.m=5 model.class_idx=[0,1] model.name_prefix=mrnet-lf-01-110 model.n_samples=83 model.train_latents=generate/latents/mrnet_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=mrnet model.m=5 model.class_idx=[0,1,2] model.name_prefix=mrnet-lf-01-111 model.n_samples=125 model.train_latents=generate/latents/mrnet_training_latents.npy


# python -m generate.generate_filtered model=generate dataset=allcts model.m=1 model.class_idx=3 model.name_prefix=allcts-051-216-gen-qs5 model.n_samples=155 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=allcts model.m=1 model.class_idx=0 model.name_prefix=allcts-051-216-gen-qs2 model.n_samples=167 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=allcts model.m=1 model.class_idx=1 model.name_prefix=allcts-051-216-gen-qs3 model.n_samples=212 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=allcts model.m=1 model.class_idx=2 model.name_prefix=allcts-051-216-gen-qs4 model.n_samples=154 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered model=generate dataset=allcts model.m=1 model.class_idx=4 model.name_prefix=allcts-051-216-gen-qs6 model.n_samples=38 #model.train_latents=generate/latents/allcts_training_latents.npy  

python -m generate.generate_filtered +experiment=gen-classifier-free-class-embedding model.m=1 model.class_idx=0 model.name_prefix=classifier-free-class-embedding-qs2 model.n_samples=10 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered +experiment=gen-classifier-free-class-embedding model.m=1 model.class_idx=0 model.name_prefix=classifier-free-class-embedding-qs2 model.n_samples=167 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered +experiment=gen-classifier-free-class-embedding model.m=1 model.class_idx=1 model.name_prefix=classifier-free-class-embedding-qs3 model.n_samples=212 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered +experiment=gen-classifier-free-class-embedding model.m=1 model.class_idx=2 model.name_prefix=classifier-free-class-embedding-qs4 model.n_samples=154 #model.train_latents=generate/latents/allcts_training_latents.npy
# python -m generate.generate_filtered +experiment=gen-classifier-free-class-embedding model.m=1 model.class_idx=4 model.name_prefix=classifier-free-class-embedding-qs6 model.n_samples=38 #model.train_latents=generate/latents/allcts_training_latents.npy
