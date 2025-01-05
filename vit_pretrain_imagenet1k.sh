#!/bin/bash

models=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
lrs=("0.0001" "0.0003" "0.001" "0.003" "0.01")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 \
      --model $model \
      --dataset imagenet \
      --data-dir ~/data/imagenet1k \
      --mean 0.485, 0.456, 0.406 --std 0.229, 0.224, 0.225 \
      --num-classes 1000 \
      --img-size 224 \
      --batch-size 1024 \
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
      --experiment IMAGENET1K
  done
done