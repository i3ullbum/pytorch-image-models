#!/bin/bash

current_time=$(date +%Y%m%d-%H%M%S)

models=("vit_tiny_patch16_224_post")
lrs=("3e-3" "1e-3" "3e-4")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
      --model $model \
      --dataset torch/cifar10 \
      --data-dir /data/rlwrld-common/beom/cifar10 \
      --dataset-download \
      --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 \
      --num-classes 10 \
      --img-size 224 \
      --batch-size 512 \
      --epochs 300 \
      --lr $lr \
      --warmup-epochs 5 \
      --weight-decay 0.05 \
      --clip-grad 1.0 \
      --sched cosine \
      --opt adamw \
      --drop-path 0.1 \
      --seed 42 \
      --log-interval 1 \
      --output /data/rlwrld-common/beom/output/cifar10/${model}_lr${lr}_${current_time} \
      --log-wandb \
      --experiment CIFAR10
  done
done
