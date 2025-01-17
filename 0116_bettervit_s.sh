#!/bin/bash

models=("vit_small_patch16_224_peri")
lrs=("0.0005")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 2 \
      --model $model \
      --dataset imagenet \
      --data-dir /data/rlwrld-common/beom/imagenet1k \
      --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
      --num-classes 1000 \
      --img-size 224 \
      --batch-size 512 \
      --epochs 90 \
      --lr $lr \
      --warmup-epochs 8 \
      --weight-decay 0.0001 \
      --clip-grad 1.0 \
      --sched cosine \
      --opt adamw \
      --mixup 0.2 --mixup-prob 1.0 --mixup-mode batch \
      --color-jitter 0.4 --train-interpolation random \
      --aa rand-m10-n2 --hflip 0.5 --reprob 0.0 \
      --train-crop-mode rrc --scale 0.08 1.0 --ratio 0.75 1.33 \
      --seed 42 \
      --log-interval 150 \
      --output ./output/better-vit-imagenet/${model}_lr${lr} \
      --log-wandb \
      --experiment BETTER-VIT-IMAGENET
  done
done