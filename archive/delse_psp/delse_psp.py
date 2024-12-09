import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101

class PSPModule(nn.Module):
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        # PSP Moduleで適切な出力チャネル数を確保
        self.stages = nn.ModuleList([self._make_stage(in_features, in_features // 4, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_features + len(sizes) * (in_features // 4), out_features, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, in_features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [x]
        for stage in self.stages:
            priors.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class DELSE_PSP(nn.Module):
    def __init__(self, num_classes):
        super(DELSE_PSP, self).__init__()
        self.backbone = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # PSP Moduleを初期レベルセット、エネルギー、ベクトル場のために用意
        self.phi_0_module = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6))
        self.energy_module = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6))
        self.vector_module = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6))

        # 各出力の最終層
        self.phi_0_final = nn.Conv2d(512, num_classes, kernel_size=1)  # phi_0出力
        self.energy_final = nn.Conv2d(512, num_classes * 2, kernel_size=1)  # エネルギーフィールド出力
        self.vector_final = nn.Conv2d(512, num_classes, kernel_size=1)  # ベクトル場出力

    def forward(self, x):
        # ResNetのバックボーンから特徴量を抽出
        x = self.backbone(x)

        # phi_0, energy, gの各モジュールを通す
        phi_0 = self.phi_0_final(self.phi_0_module(x))  # 初期レベルセット
        energy = self.energy_final(self.energy_module(x))  # エネルギーフィールド
        vector_field = self.vector_final(self.vector_module(x))  # ベクトル場

        return phi_0, energy, vector_field
