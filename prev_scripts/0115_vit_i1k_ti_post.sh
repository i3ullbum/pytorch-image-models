#!/bin/bash

export OMP_NUM_THREADS=2

models=("vit_tiny_patch16_224_post")
lrs=("3e-3")
# lrs=("1e-4" "3e-4" "1e-3" "3e-3" "1e-2")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 2 \
      --model $model \
      --dataset imagenet \
      --data-dir /data/rlwrld-common/beom/imagenet1k \
      --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
      --num-classes 1000 \
      --img-size 224 \
      --batch-size 2048 \
      --epochs 300 \
      --lr $lr \
      --warmup-epochs 5 \
      --weight-decay 0.3 \
      --clip-grad 1.0 \
      --smoothing 0.1 \
      --sched cosine \
      --opt adamw \
      --drop-path 0.1 \
      --seed 7777 \
      --log-interval 20 \
      --output ./output/imagenet1k/${model}_lr${lr} \
      --log-wandb \
      --experiment IMAGENET1K \
      --workers 16 \
      --pin-mem
  done
done