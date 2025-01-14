import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    シンプルな畳み込みブロック: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, args):
        super(UNetPlusPlus, self).__init__()
        base_filters = 64
        in_channels = args.num_channels
        out_channels = args.num_classes
        
        # Encoder
        self.conv0_0 = ConvBlock(in_channels, base_filters)                       # (C -> 64)
        self.conv1_0 = ConvBlock(base_filters, base_filters * 2)                  # (64 -> 128)
        self.conv2_0 = ConvBlock(base_filters * 2, base_filters * 4)              # (128 -> 256)
        self.conv3_0 = ConvBlock(base_filters * 4, base_filters * 8)              # (256 -> 512)

        # Decoder
        self.conv0_1 = ConvBlock(base_filters + base_filters, base_filters)       # (64 + 64 -> 64)
        self.conv1_1 = ConvBlock(base_filters * 2 + base_filters * 2, base_filters * 2)  # (128 + 256 -> 128)
        self.conv2_1 = ConvBlock(base_filters * 4 + base_filters * 4, base_filters * 4)  # (256 + 512 -> 256)

        self.conv0_2 = ConvBlock(base_filters * 2, base_filters)                  # (64 + 64 -> 64)
        self.conv1_2 = ConvBlock(base_filters * 2 + base_filters * 2, base_filters * 2)  # (128 + 256 -> 128)

        self.conv0_3 = ConvBlock(base_filters * 2, base_filters)                  # (64 + 64 -> 64)

        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling
        self.up1_0 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)

        self.up1_2 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)

        # Final Convolution
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.direction = nn.Conv2d(base_filters, 2, kernel_size=9, padding=4)

    def forward(self, x):
        # Encoder
        with torch.no_grad():
            x0_0 = self.conv0_0(x)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x2_0 = self.conv2_0(self.pool(x1_0))
            x3_0 = self.conv3_0(self.pool(x2_0))
        
            # Decoder
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        
            x0_2 = self.conv0_2(torch.cat([x0_1, self.up1_1(x1_1)], dim=1))
            x1_2 = self.conv1_2(torch.cat([x1_1, self.up2_1(x2_1)], dim=1))
        
            x0_3 = self.conv0_3(torch.cat([x0_2, self.up1_2(x1_2)], dim=1))
        
            # Output
            out = self.final_conv(x0_3)
        direction = self.direction(x0_3)
        
        return out, direction
