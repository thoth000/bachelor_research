#!/bin/bash

# OpenMPのスレッド数を1に設定して、負荷を削減する
export OMP_NUM_THREADS=1

# エラー回避
export NO_ALBUMENTATIONS_UPDATE=1

# マスターポートの設定
MASTER_PORT=$((50000 + RANDOM % 1000))  # 50000〜50999の範囲でランダムなポート番号

# 使用するGPUを指定する（例：4 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# モデル名
MODEL_NAME="unet"
# トレーニング用引数の指定
MAX_EPOCH=40
BATCH_SIZE=4
RESOLUTION=584
LEARNING_RATE=1e-4
ETA_MIN=0
WEIGHT_DECAY=1e-5
CRITERION="BCE"

SCHEDULER="cosine_annealing"
EXP_DIR="exp"
EXP_NAME="exp_$(date +"%Y%m%d_%H%M%S")"  # exp_nameをタイムスタンプに基づいて設定
VAL_INTERVAL=1
THRESHOLD=0.5
NUM_WORKERS=4
DATASET="drive"
TRANSFORM="fr_unet"
DATASET_PATH="/home/sano/dataset/DRIVE"
DATASET_OPT="128_RV-GAN"
PRETRAINED_PATH="/home/sano/documents/fr_unet_improved_increase/models/checkpoint-epoch40.pth"

ALPHA=0.5
EXP_NAME="dice_cldice_alpha0.5"

# PyTorch DDPでトレーニングを実行
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port=$MASTER_PORT main.py \
    --model_name $MODEL_NAME \
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
    --threshold $THRESHOLD \
    --num_workers $NUM_WORKERS \
    --dataset $DATASET \
    --transform $TRANSFORM \
    --dataset_path $DATASET_PATH \
    --dataset_opt $DATASET_OPT \
    --save_mask \
    --pretrained_path $PRETRAINED_PATH \
    --alpha $ALPHA
