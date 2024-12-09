import torch

config = {
    # データ系
    "dataset": "cityscapes",
    "train_data": "/home/sano/dataset/cityscapes/leftImg8bit",
    "val_data": "/home/sano/dataset/cityscapes/leftImg8bit",
    "test_data": "/home/sano/dataset/cityscapes/leftImg8bit", 
    "num_classes": 19,  # CityScapesのクラス数
    # 学習パラメータ
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "epochs": 100,
    "num_workers": 4,
    "optimizer": "SGD",   # Default optimizer
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "/home/sano/documents/segmentation/pspnet/exp/pspnet_epoch100/final_model.pth",  # テスト時に使用するモデルのパス
    # 実行系
    "mode": "train",  # 実行モード: train または test
    "eval_interval": 1,  # 何エポックごとに評価
    "output_dir": "/home/sano/documents/segmentation/pspnet/result"
}
