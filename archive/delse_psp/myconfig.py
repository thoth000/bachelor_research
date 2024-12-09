import torch
from dataloaders.class_mapping import get_num_classes

config = {
    # データ系
    "dataset": "cityscapes", # cityscapes_car
    "train_data": "/home/sano/dataset/cityscapes/leftImg8bit",
    "val_data": "/home/sano/dataset/cityscapes/leftImg8bit",
    "test_data": "/home/sano/dataset/cityscapes/leftImg8bit", 
    
    # 学習パラメータ
    "lr": 3e-4,
    "factor": 0.1,
    "patience": 5,
    "threshold": 1e-4,
    "cooldown": 2,
    "min_lr": 1e-6,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "batch_size": 16,
    "epochs": 200,
    "num_workers": 4,
    "optimizer": "SGD",   # Default optimizer
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": None,
    
    # レベルセット関連のパラメータ
    'T': 5,  # レベルセット進化のステップ数
    'epsilon': -1,  # Heavisideの平滑化
    'dt_max': 30,  # 最大時間ステップ
    'alpha': 1,  # level_map_loss用の重み
    'sigma': 0.08,  # 符号付距離の平滑化
    'distance_max': 30,  # 符号付距離の上限
    'random_shift_range': 10,  # 初期レベルセットへのランダムシフト範囲
    
    # トレーニングの設定
    'pretrain_epoch': 5,  # プリトレーニングのエポック数
    'e2e': False,  # エンドツーエンド学習の有無
    
    # 実行系
    "mode": "train",  # 実行モード: train または test
    # "eval_interval": 5,  # 何エポックごとに評価
    "output_dir": "/home/sano/documents/segmentation/delse_psp/result"
}
