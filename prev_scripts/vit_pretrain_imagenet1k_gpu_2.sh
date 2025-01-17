#!/bin/bash

current_time=$(date +%Y%m%d-%H%M%S)


models=("vit_small_patch16_224_peri")
lrs=("3e-3" "1e-3" "3e-4")


for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 2 \
      --model $model \
      --dataset imagenet \
      --data-dir /data/rlwrld-common/beom/imagenet1k \
      --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
      --num-classes 1000 \
      --img-size 224 \
      --batch-size 1024 --grad-accum-steps 2 \
      --epochs 300 \
      --lr $lr \
      --warmup-epochs 5 \
      --weight-decay 0.3 \
      --clip-grad 1.0 \
      --sched cosine \
      --opt adamw \
      --drop-path 0.1 \
      --seed 42 \
      --log-interval 1 \
      --output ./output/imagenet1k/${model}_lr${lr}_${current_time} \
      --log-wandb \
      --experiment IMAGENET1K
  done
done
