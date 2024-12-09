#!/bin/bash

# エラー回避
export NO_ALBUMENTATIONS_UPDATE=1

# マスターポートの設定
MASTER_PORT=$((50000 + RANDOM % 1000))  # 50000〜50999の範囲でランダムなポート番号

# 使用するGPUを指定する（例：4 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# トレーニング用引数の指定
MAX_EPOCH=5000
BATCH_SIZE=12
RESOLUTION=672
LEARNING_RATE=1e-4
ETA_MIN=1e-6
WEIGHT_DECAY=1e-5
CRITERION="BCE"
SCHEDULER="cosine_annealing"
EXP_DIR="exp"
VAL_INTERVAL=10
PRETRAINED_PATH="/home/sano/documents/swin_unet_drive/models/swin_tiny_patch4_window7_224.pth"

# 実行時間のタイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="exp_$TIMESTAMP"  # exp_nameをタイムスタンプに基づいて設定

# PyTorch DDPでトレーニングを実行
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port=$MASTER_PORT main.py \
    --max_epoch $MAX_EPOCH \
    --batch $BATCH_SIZE \
    --resolution $RESOLUTION \
    --lr $LEARNING_RATE \
    --eta_min $ETA_MIN \
    --weight_decay $WEIGHT_DECAY \
    --criterion $CRITERION \
    --scheduler $SCHEDULER \
    --exp_dir $EXP_DIR \
    --exp_name $EXP_NAME \
    --val_interval $VAL_INTERVAL \
    --pretrained_path $PRETRAINED_PATH
