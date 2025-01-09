#!/bin/bash

current_time=$(date +%Y%m%d-%H%M%S)

models=("vit_small_patch16_224_post" "vit_small_patch16_224_peri" "vit_small_patch16_224")
lrs=("3e-3" "1e-3" "3e-4")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 2 \
      --model $model \
      --dataset torch/cifar10 \
      --data-dir /data/rlwrld-common/beom/cifar10 \
      --dataset-download \
      --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 \
      --num-classes 10 \
      --img-size 224 \
      --batch-size 256 \
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
      --output ./output/cifar10/${model}_lr${lr}_${current_time} \
      --log-wandb \
      --experiment CIFAR10
  done
done
