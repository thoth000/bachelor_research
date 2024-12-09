import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.backbone = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.ppm = nn.ModuleList([self._make_ppm_layer(2048, 512, pool_size) for pool_size in [1, 2, 3, 6]])
        self.conv_last = nn.Conv2d(4096, 512, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.aux_classifier = nn.Conv2d(2048, num_classes, kernel_size=1)

    def _make_ppm_layer(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        input_size = x.size()
        x = self.backbone(x)
        # 補助出力
        aux_out = self.aux_classifier(x)

        ppm_outs = [x]
        for ppm in self.ppm:
            ppm_outs.append(F.interpolate(ppm(x), size=x.size()[2:], mode='bilinear', align_corners=True))
        x = torch.cat(ppm_outs, dim=1)

        x = self.conv_last(x)
        main_out = F.interpolate(self.classifier(x), size=input_size[2:], mode='bilinear', align_corners=True)

        return main_out, aux_out
