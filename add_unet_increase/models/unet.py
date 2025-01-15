import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------------------------------
# DoubleConv
#-----------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    2つの連続した3x3畳み込み + BatchNorm + 活性化関数 + Dropout を含むモジュール
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.0, activation='relu'):
        super(DoubleConv, self).__init__()

        # 活性化関数を動的に選択
        if activation.lower() == 'relu':
            act_layer = nn.ReLU
        elif activation.lower() == 'leakyrelu':
            # 例としてnegative_slope=0.1に
            act_layer = lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation.lower() == 'elu':
            act_layer = nn.ELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # BatchNorm追加
            act_layer(inplace=True),
            nn.Dropout2d(p=dropout_p),     # Dropout追加

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # BatchNorm追加
            act_layer(inplace=True),
            nn.Dropout2d(p=dropout_p)      # Dropout追加
        )

    def forward(self, x):
        return self.double_conv(x)

#-----------------------------------------------------------------
# Down
#-----------------------------------------------------------------
class Down(nn.Module):
    """
    ダウンサンプリングブロック:
      - 最大プーリング + DoubleConv
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.0, activation='relu'):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p, activation=activation)
        )

    def forward(self, x):
        return self.down(x)

#-----------------------------------------------------------------
# Up
#-----------------------------------------------------------------
class Up(nn.Module):
    """
    アップサンプリングブロック:
      - ConvTranspose2d で2倍拡大
      - サイズを合わせてスキップ接続
      - DoubleConv
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.0, activation='relu'):
        super(Up, self).__init__()
        # オリジナル同様に転置畳み込みを使用
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p, activation=activation)

    def forward(self, x1, x2):
        # 転置畳み込みでアップサンプリング
        x1 = self.up(x1)
        
        # サイズを合わせるためにパディングを調整
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # スキップ接続
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-----------------------------------------------------------------
# UNet
#-----------------------------------------------------------------
class UNet(nn.Module):
    """
    オリジナルのU-Net構造 (ダウンサンプリング4回, アップサンプリング4回)
    """
    def __init__(self, args):
        """
        Args:
            in_channels (int): 入力画像チャネル数
            out_channels (int): 出力クラス数 (2クラスなど)
            dropout_p (float): Dropoutの確率
            activation (str): 活性化関数の種類 ('relu', 'leakyrelu', 'elu' など)
        """
        super(UNet, self).__init__()

        in_channels = args.num_channels
        out_channels = args.num_classes
        dropout_p = args.dropout_p
        activation = args.activation

        # Down
        self.in_conv = DoubleConv(in_channels, 64, dropout_p=dropout_p, activation=activation)
        self.down1 = Down(64, 128, dropout_p=dropout_p, activation=activation)
        self.down2 = Down(128, 256, dropout_p=dropout_p, activation=activation)
        self.down3 = Down(256, 512, dropout_p=dropout_p, activation=activation)
        self.down4 = Down(512, 1024, dropout_p=dropout_p, activation=activation)

        # Up
        self.up1 = Up(1024, 512, dropout_p=dropout_p, activation=activation)
        self.up2 = Up(512, 256, dropout_p=dropout_p, activation=activation)
        self.up3 = Up(256, 128, dropout_p=dropout_p, activation=activation)
        self.up4 = Up(128, 64, dropout_p=dropout_p, activation=activation)

        # 出力層
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # 方向推定層
        self.direction = nn.Conv2d(64, 2, kernel_size=9, padding=4)

    def forward(self, x):
        # Down
        with torch.no_grad():
            x1 = self.in_conv(x)     # (N, 64, H, W)
            x2 = self.down1(x1)      # (N, 128, H/2, W/2)
            x3 = self.down2(x2)      # (N, 256, H/4, W/4)
            x4 = self.down3(x3)      # (N, 512, H/8, W/8)
            x5 = self.down4(x4)      # (N, 1024, H/16, W/16)

            # Up
            x = self.up1(x5, x4)     # (N, 512, H/8, W/8)
            x = self.up2(x, x3)      # (N, 256, H/4, W/4)
            x = self.up3(x, x2)      # (N, 128, H/2, W/2)
            x = self.up4(x, x1)      # (N, 64, H, W)

            # 出力
            out = self.out_conv(x)     # (N, out_channels, H, W)

        direction = self.direction(x) # (N, 2, H, W)
        return out, direction


#-----------------------------------------------------------------
# 動作確認
#-----------------------------------------------------------------
if __name__ == "__main__":
    # サンプル入力 (batch=1, RGB, 256x256)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Dropout=0.1, 活性化関数='elu' のUNetを例示
    model = UNet(in_channels=3, out_channels=2, dropout_p=0.1, activation='elu')
    output = model(dummy_input)
    print("出力サイズ:", output.shape)  # 期待: (1, 2, 256, 256)
