import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import InitWeights_He


class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
        

    def forward(self, x):
        return self.conv(x)


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dp=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class FR_UNet(nn.Module):
    def __init__(self,  args):
        super(FR_UNet, self).__init__()
        # 引数の設定
        num_classes = args.num_classes
        num_channels = 1 # args.num_channels
        feature_scale = args.feature_scale
        dropout = args.dropout
        fuse = args.fuse
         # self.out_ave = args.out_ave
        
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        output = (self.final1(x1_1)+self.final2(x10)+self.final3(x11)+self.final4(x12)+self.final5(x13)) / 5
        #if self.out_ave == True:
        #    output = (self.final1(x1_1)+self.final2(x10) +
        #              self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        #else:
        #    output = self.final5(x13)

        return output

class Allen_Cahn_Model(nn.Module):
    def __init__(self, args):
        super(Allen_Cahn_Model, self).__init__()
        self.model = FR_UNet(args)
        # pde parameters
        self.gamma_row = nn.Parameter(torch.tensor(-5.0))
        self.M = args.M # 移動度
        self.dt = args.dt # 時間刻み
        self.steps = args.steps # ステップ数
    
    def gradient(self, u):
        # ノイマン境界条件下での勾配計算
        # u (torch.Tensor): (B, C, H, W)
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
        dx = (u_pad[..., 2:, 1:-1] - u_pad[..., :-2, 1:-1]) / 2
        dy = (u_pad[..., 1:-1, 2:] - u_pad[..., 1:-1, :-2]) / 2
        return dx, dy
    
    def divergence(self, dx, dy):
        # ノイマン境界条件下での発散計算
        # dx, dy (torch.Tensor): (B, C, H, W)
        dx_pad = F.pad(dx, (1, 1, 1, 1), mode='replicate')
        dy_pad = F.pad(dy, (1, 1, 1, 1), mode='replicate')
        ddx = (dx_pad[..., 2:, 1:-1] - dx_pad[..., :-2, 1:-1]) / 2
        ddy = (dy_pad[..., 1:-1, 2:] - dy_pad[..., 1:-1, :-2]) / 2
        return ddx + ddy
    
    def allen_cahn(self, u):
        # Allen-Cahn方程式の計算
        # u (torch.Tensor): (B, C, H, W)
        # du = M * {div(gamma * grad(u)) - u(1-u)(2u-1)}
        dx, dy = self.gradient(u)
        gamma = torch.exp(self.gamma_row) # expで正の値を保証
        laplacian = self.divergence(gamma * dx, gamma * dy)
        return self.M * (laplacian - (u*(1-u)*(2*u-1)))
        
    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        # Allen-Cahn equation
        for _ in range(self.steps):
            # Euler method
            #x = x + self.dt * self.allen_cahn(x)
            
            # Runge-Kutta method 4th order
            k1 = self.allen_cahn(x)
            k2 = self.allen_cahn(x + self.dt/2 * k1)
            k3 = self.allen_cahn(x + self.dt/2 * k2)
            k4 = self.allen_cahn(x + self.dt * k3)
            x = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
        # 相分離モデル化のために[0, 1]にクリッピング
        x = torch.clamp(x, 0, 1)
        return x