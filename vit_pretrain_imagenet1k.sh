#!/bin/bash

models=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
lrs=("3e-3")
# lrs=("1e-4" "3e-4" "1e-3" "3e-3" "1e-2")


for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 8 \
      --model $model \
      --dataset imagenet \
      --data-dir ~/data/imagenet1k \
      --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
      --num-classes 1000 \
      --img-size 224 \
      --batch-size 512 \ # 4096/8
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
