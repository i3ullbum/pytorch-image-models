#!/bin/bash
#SBATCH --job-name=imagenet_train      # A descriptive name for your job
#SBATCH --partition=amd_a100nv_8                # The GPU partition/queue name
#SBATCH --gres=gpu:8                   # Request 8 GPUs (adapt if you only have 2 or 4 per node)
#SBATCH --nodes=1                      # Number of nodes (1 node here; if you need multiple nodes, increase this)
#SBATCH --cpus-per-task=32            # Example CPU count; adapt to your needs
#SBATCH --mem=128G                     # System memory; adapt to your needs
#SBATCH --time=48:00:00                # Wall time limit; e.g., 48 hours
#SBATCH --comment=pytorch             # REQUIRED: "pytorch" is accepted on your cluster
#SBATCH --output=imagenet_%j.out       # Standard output file
#SBATCH --error=imagenet_%j.err        # Standard error file

# (1) Load Modules or Conda Environment
# Adjust these module/conda commands to match your environment
module purge
conda activate vitt  # or your environment name

# (2) Go to the code directory
cds
cd /pytorch-image-models

# (3) Run your training loops
models=("vit_tiny_patch16_224_post" "vit_small_patch16_224_post" "vit_base_patch16_224_post")
lrs=("3e-3")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 8 \
      --model $model \
      --dataset imagenet \
      --data-dir /scratch/x3135a05/imagenet1k \
      --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
      --num-classes 1000 \
      --img-size 224 \
      --batch-size 512 \
      --epochs 300 \
      --lr $lr \
      --warmup-epochs 5 \
      --weight-decay 0.3 \
      --clip-grad 1.0 \
      --smoothing 0.1 \
      --sched cosine \
      --opt adamw \
      --drop-path 0.1 \
      --seed 1234 \
      --log-interval 1 \
      --output ./output/imagenet1k/${model}_lr${lr} \
      --log-wandb \
      --experiment IMAGENET1K
  done
done
