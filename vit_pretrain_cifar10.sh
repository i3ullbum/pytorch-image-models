#!/bin/bash

models=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
lrs=("3e-3")
# lrs=("1e-4" "3e-4" "1e-3" "3e-3" "1e-2")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 8 \
      --model $model \
      --dataset torch/cifar10 \
      --data-dir ~/data/cifar10 \
      --dataset-download \
      --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 \
      --num-classes 10 \
      --img-size 224 \
      --batch-size 1024 \
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
      --output ./output/cifar10/${model}_lr${lr} \
      --log-wandb \
      --experiment CIFAR10
  done
done
