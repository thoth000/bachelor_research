#!/bin/bash

# エラー回避
export NO_ALBUMENTATIONS_UPDATE=1

# システムの負荷を下げるためにスレッド数を制限する
OMP_NUM_THREADS=1

# マスターポートの設定
MASTER_PORT=$((50000 + RANDOM % 1000))  # 50000〜50999の範囲でランダムなポート番号

# 使用するGPUを指定する（例：4 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

BATCH_SIZE=1
RESOLUTION=584
THRESHOLD=0
SAVE_DIR_ROOT="result"
DATASET_PATH="/home/sano/dataset/DRIVE"
DATASET_OPT="560"
PRETRAINED_PATH="/home/sano/documents/cahn_hilliard_FR-UNet/models/checkpoint-epoch40.pth"
pde=7
# パラメータのリスト
D_VALUES=(1.0)
GAMMA_VALUES=(1.0 0.5 0.25)
DT_VALUES=(0.01)
STEPS_VALUES=(5000)

# 全ての組み合わせを試す
for D in "${D_VALUES[@]}"; do
    for GAMMA in "${GAMMA_VALUES[@]}"; do
        for DT in "${DT_VALUES[@]}"; do
            for STEPS in "${STEPS_VALUES[@]}"; do

                # 保存ディレクトリ名を設定
                SAVE_NAME="PDE${pde}_GAMMA${GAMMA}"
                clear
                echo "SAVE_NAME: $SAVE_NAME"

                # 実行
                torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port=$MASTER_PORT test.py \
                    --batch $BATCH_SIZE \
                    --threshold $THRESHOLD \
                    --save_dir_root $SAVE_DIR_ROOT \
                    --save_name $SAVE_NAME \
                    --dataset_path $DATASET_PATH \
                    --dataset_opt $DATASET_OPT \
                    --pretrained_path $PRETRAINED_PATH \
                    --save_mask \
                    --pde $pde \
                    --D $D \
                    --gamma $GAMMA \
                    --dt $DT \
                    --steps $STEPS

            done
        done
    done
done
