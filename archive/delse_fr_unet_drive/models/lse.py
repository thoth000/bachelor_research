from __future__ import division
import numpy as np
import cv2
from scipy import misc, ndimage
import torch
from torch.nn import functional as F

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

def del2(x):
    assert x.dim() == 4
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = torch.FloatTensor(laplacian).unsqueeze(0).unsqueeze(0).cuda()
    x = F.conv2d(x, laplacian, padding=0)
    x = torch.nn.ReplicationPad2d(1)(x)
    return x

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
        return torch.cat((gy, gx), dim=1)
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
    g = f.clone()
    g = torch.reshape(g, (N * K, H, W))
    [_, nrow, ncol] = g.shape
    g[..., [0, nrow - 1], [0, ncol - 1]] = g[..., [2, nrow - 3], [2, ncol - 3]]
    g[..., [0, nrow - 1], 1: ncol - 1] = g[..., [2, nrow - 3], 1: ncol - 1]
    g[..., 1: nrow - 1, [0, ncol - 1]] = g[..., 1: nrow - 1, [2, ncol - 3]]
    g = torch.reshape(g, (N, K, H, W))
    return g

def levelset_evolution(phi, vx, vy, m=None, T=5, timestep=5, dirac_ratio=0.3, dt_max=30, _normalize=True):
    if _normalize:
        norm = torch.sqrt(vx ** 2 + vy ** 2 + 1e-10)
        vx = vx / norm
        vy = vy / norm

    for k in range(T):
        phi = NeumannBoundCond(phi)
        phi_y, phi_x = gradient_sobel(phi, split=True)
        s = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-10)
        Nx = phi_x / s
        Ny = phi_y / s
        curvature = div(Nx, Ny)
        diracPhi = Dirac(phi, dirac_ratio, dt_max=dt_max)
        motion_term = (vx * phi_x + vy * phi_y) * -1 # mortion termに-1を掛けるべき
        

        if m is not None:
            phi = phi + timestep * diracPhi * (motion_term + m * curvature) # 外部項 + 内部項(選択的形状正則化)
        else:
            phi = phi + timestep * diracPhi * (motion_term + m * curvature)

        phi = phi + 0.2 * distReg_p2(phi) # 正則化
    return phi
