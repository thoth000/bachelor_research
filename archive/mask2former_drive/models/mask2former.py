import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Mask2FormerForUniversalSegmentation
from huggingface_hub import login

from .utils import read_mask2former_token

class Mask2FormerSegmentation(nn.Module):
    def __init__(self, args):
        super(Mask2FormerSegmentation, self).__init__()
        # 事前学習済モデルの読み込み
        login(read_mask2former_token(token_file_path=args.token_path))
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(args.pretrained_path)
        # セグメンテーションヘッド (最終出力に使用する部分)
        self.classifier = nn.Conv2d(100, args.num_classes, kernel_size=1)  # num_queries = 100を想定

    def forward(self, x):
        input_size = x.shape[-2:]
        # Mask2Formerの事前処理
        mask2former_outputs = self.mask2former(x)
        masks_queries_logits = F.interpolate(mask2former_outputs.masks_queries_logits, size=input_size, mode='bilinear', align_corners=False) # [batch, num_queries, height, width]
        # Conv2dを通してクラスごとのlogitsに変換
        logits = self.classifier(masks_queries_logits)  # [batch_size, num_classes, height, width]

        return logits

# 使用例
if __name__ == "__main__":
    # モデルの作成
    model = Mask2FormerSegmentation(num_classes=21)  # 21クラスの場合 (PASCAL VOCなど)

    # ダミーデータの作成 (バッチサイズ1、256x256の画像)
    input_tensor = torch.randn(1, 3, 256, 256)

    # 推論
    output = model(input_tensor)

    # 出力の形状を確認
    print(output.shape)  # [batch, num_classes, H, W]の形式になる
