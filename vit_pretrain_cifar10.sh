#!/bin/bash

models=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
lrs=("0.0001" "0.0003" "0.001" "0.003" "0.01")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 \
      --model $model \
      --dataset torch/cifar10 \
      --data-dir ~/data/cifar10 \
      --dataset-download \
      --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 \
      --num-classes 10 \
      --img-size 224 \
      --batch-size 256 \
      --epochs 300 \
      --lr $lr \
      --warmup-epochs 10 \
      --weight-decay 0.1 \
      --mixup 0.8 \
      --cutmix 1.0 \
      --smoothing 0.1 \
      --sched cosine \
      --opt adamw \
      --drop-path 0.1 \
      --seed 1234 \
      --log-interval 1 \
      --output ./output/${model}_lr${lr} \
      --log-wandb \
      --experiment CIFAR10
  done
done