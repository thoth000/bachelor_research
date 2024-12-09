import torch
import numpy as np

def Heaviside(v, epsilon=-1/2):
    pi = 3.141593
    v = 0.5 * (1 + 2/pi * torch.atan(v/epsilon))
    return v

def Dirac(x, sigma=0.2, dt_max=30):
    '''This is an adapted version of Dirac function'''
    x = x / dt_max
    f = (1.0 / 2.0) * (1 + torch.cos(np.pi * x / sigma))
    f[(x >= sigma) | (x <= -sigma)] = 0
    return f

def gradient(x, split=True):
    [nrow, ncol] = x.shape[-2:]

    gy = x.clone()
    gy[..., 1:nrow - 1, :] = (x[..., 2:nrow, :] - x[..., 0:nrow - 2, :]) / 2
    gy[..., 0, :] = x[..., 1, :] - x[..., 0, :]
    gy[..., nrow - 1, :] = x[..., nrow - 1, :] - x[..., nrow - 2, :]

    gx = x.clone()
    gx[..., 1:ncol - 1] = (x[..., 2:ncol] - x[..., 0:ncol - 2]) / 2
    gx[..., 0] = x[..., 1] - x[..., 0]
    gx[..., ncol - 1] = x[..., ncol - 1] - x[..., ncol - 2]

    if not split:
        return torch.cat([gx.unsqueeze(1), gy.unsqueeze(1)], dim=1)
    return gy, gx

def gradient_sobel(map, split=True):
    return gradient(map, split)

def div(nx, ny):
    [_, nxx] = gradient_sobel(nx, split=True)
    [nyy, _] = gradient_sobel(ny, split=True)
    return nxx + nyy

def distReg_p2(phi):
    [phi_y, phi_x] = gradient_sobel(phi, split=True)
    s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
    a = ((s >= 0) & (s <= 1)).float()
    b = (s > 1).float()
    ps = a * torch.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    neq0 = lambda x: ((x < -1e-10) | (x > 1e-10)).float()
    eq0 = lambda x: ((x >= -1e-10) & (x <= 1e-10)).float()
    dps = (neq0(ps) * ps + eq0(ps)) / (neq0(s) * s + eq0(s))
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + del2(phi)

def NeumannBoundCond(f):
    N, K, H, W = f.shape
    g = f  
    g = torch.reshape(g, (N * K, H, W))
    [_, nrow, ncol] = g.shape
    g[..., [0, nrow - 1], [0, ncol - 1]] = g[..., [2, nrow - 3], [2, ncol - 3]]
    g[..., [0, nrow - 1], 1: ncol - 1] = g[..., [2, nrow - 3], 1: ncol - 1]
    g[..., 1: nrow - 1, [0, ncol - 1]] = g[..., 1: nrow - 1, [2, ncol - 3]]
    g = torch.reshape(g, (N, K, H, W))
    return g

def levelset_evolution(phi, energy, g, T=5, dt_max=30, **kwargs):
    """
    レベルセット進化の処理
    phi: 初期レベルセット
    energy: ベクトル場 (batch_size, 2, num_classes, H, W)
    g: スカラーのエネルギーマップ
    T: 進化のステップ数
    dt_max: 最大ステップ幅
    """

    vx = energy[:, 0, :, :, :]  # ベクトル場の x 方向成分
    vy = energy[:, 1, :, :, :]  # ベクトル場の y 方向成分

    for t in range(T):
        phi = NeumannBoundCond(phi)  # Neumann境界条件を適用
        phi_x, phi_y = gradient(phi)  # レベルセットの勾配

        s = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-10)  # 勾配の大きさ
        Nx = phi_x / s  # 法線ベクトルのx成分
        Ny = phi_y / s  # 法線ベクトルのy成分
        curvature = div(Nx, Ny)  # 曲率計算

        # Dirac Delta関数の適用
        dirac_phi = Dirac(phi)

        # モーション項の計算 (ベクトル場を使用)
        motion_term = vx * phi_x + vy * phi_y

        # レベルセットの更新 (スカラー場gを使って曲率を制御)
        phi = phi + dt_max * dirac_phi * (motion_term + g * curvature)
    
    return phi
