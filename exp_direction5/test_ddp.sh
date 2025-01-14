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
MODEL_NAME="fr_unet"

# トレーニング用引数の指定
MAX_EPOCH=200
BATCH_SIZE=1
RESOLUTION=768
LEARNING_RATE=1e-2
ETA_MIN=0
WEIGHT_DECAY=1e-5
CRITERION="Dice"
SCHEDULER="cosine_annealing"
EXP_DIR="exp"
EXP_NAME="exp_$(date +"%Y%m%d_%H%M%S")"  # exp_nameをタイムスタンプに基づいて設定
VAL_INTERVAL=5
THRESHOLD=0.5
NUM_WORKERS=4
DATASET="drive"
TRANSFORM="fr_unet"
DATASET_PATH="/home/sano/dataset/DRIVE"
DATASET_OPT="560"
PRETRAINED_PATH="/home/sano/documents/exp_direction/exp/0_1.0_0_0/final_model.pth"

# Lossの重みを指定
ALPHA=0
LAMBDA_MAIN=1.0
LAMBDA_COSINE=0
LAMBDA_ANISOTROPIC=0 

# PyTorch DDPでトレーニングを実行
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port=$MASTER_PORT test.py \
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
    --pretrained_path $PRETRAINED_PATH \
    --fix_pretrained_params True \
    --save_mask \
    --alpha $ALPHA \
    --lambda_main $LAMBDA_MAIN \
    --lambda_cosine $LAMBDA_COSINE \
    --lambda_anisotropic $LAMBDA_ANISOTROPIC