#!/bin/bash

# エラー回避
export NO_ALBUMENTATIONS_UPDATE=1

# マスターポートの設定
MASTER_PORT=$((50000 + RANDOM % 1000))  # 50000〜50999の範囲でランダムなポート番号

# 使用するGPUを指定する（例：4 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# トレーニングと共通するパラメータ
BATCH_SIZE=6
RESOLUTION=584
THRESHOLD=0.5
RESULT_DIR="result"
DATASET_PATH="/home/sano/dataset/DRIVE"
DATASET_OPT="pad"
PRETRAINED_PATH="/home/sano/documents/exp_unet/exp/AdamW_48/final_model.pth"

# 実行時間のタイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_NAME="result_$TIMESTAMP"  # テスト結果ディレクトリをタイムスタンプに基づいて設定

# PyTorch DDPでテストを実行
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port=$MASTER_PORT test.py \
    --batch $BATCH_SIZE \
    --resolution $RESOLUTION \
    --threshold $THRESHOLD \
    --result_dir $RESULT_DIR \
    --result_name $RESULT_NAME \
    --dataset_path $DATASET_PATH \
    --dataset_opt $DATASET_OPT \
    --pretrained_path $PRETRAINED_PATH
