#!/bin/bash

current_time=$(date +%Y%m%d-%H%M%S)


models=("vit_small_patch16_224")
lrs=("3e-3")
weightdecays=("0.3" "0.1" "0.05" "0.01")


for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    for weightdecay in "${weightdecays[@]}"; do
      ./distributed_train.sh 2 \
        --model $model \
        --dataset imagenet \
        --data-dir /data/rlwrld-common/beom/imagenet1k \
        --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
        --num-classes 1000 \
        --img-size 224 \
        --batch-size 1024 --grad-accum-steps 2 \
        --epochs 70 \
        --lr $lr \
        --warmup-epochs 5 \
        --weight-decay $weightdecay \
        --clip-grad 1.0 \
        --sched cosine \
        --opt adamw \
        --drop-path 0.1 \
        --seed 42 \
        --log-interval 100 \
        --output ./output/imagenet1k-ti-search/${model}_lr${lr}_${current_time} \
        --log-wandb \
        --experiment IMAGENET1K-TI-SEARCH
    done
  done
done