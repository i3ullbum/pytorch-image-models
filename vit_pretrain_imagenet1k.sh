models=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
lrs=("0.0001" "0.0003" "0.001" "0.003" "0.01")

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    ./distributed_train.sh 1 \
      --model $model \
      --data-dir /scratch/x3135a05/imagenet1k/ \
      --epochs 300 \
      --lr $lr \
      --warmup-epochs 10 \
      --weight-decay 0.1 \
      --mixup 0.8 \
      --cutmix 1.0 \
      --smoothing 0.1 \
      --sched cosine \
      --opt adamw \
      --batch-size 1024 \
      --drop-path 0.1 \
      --seed 42 \
      --log-interval 30 \
      --output ./output/${model}_lr${lr} \
      --log-wandb
  done
done