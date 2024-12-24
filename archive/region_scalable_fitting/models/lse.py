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
    """
    Applies Neumann boundary condition to a 2D or 4D tensor.
    
    Args:
        f (torch.Tensor): Input tensor (N, 1, H, W).
        
    Returns:
        torch.Tensor: Tensor with Neumann boundary conditions applied.
    """
    N, C, H, W = f.shape
    g = f.clone()

    # Top and bottom boundaries
    g[:, :, 0, :] = g[:, :, 1, :]
    g[:, :, H-1, :] = g[:, :, H-2, :]

    # Left and right boundaries
    g[:, :, :, 0] = g[:, :, :, 1]
    g[:, :, :, W-1] = g[:, :, :, W-2]

    return g

def g(h, c=1.0):
    """
    Edge-stopping function g(h) = 1 / (1 + c * h^2).
    Args:
        h (torch.Tensor): Gradient magnitude |∇I|.
        c (float): Scaling parameter (default=1.0).
    Returns:
        torch.Tensor: Edge-stopping function values.
    """
    return 1 / (1 + c * h**2)

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


def levelset_evolution_pde(phi, I, P, alpha=0.6, c1=1.0, c2=-0.1, c_g=1.0, T=5, timestep=5, dirac_ratio=0.3, dt_max=30):
    """
    Implements the level set evolution based on the given PDE for 2D images.

    Args:
        phi (torch.Tensor): Level set function (N, 1, H, W).
        I (torch.Tensor): Input image (N, 1, H, W).
        P (torch.Tensor): Vessel region probability (N, 1, H, W).
        alpha (float): Weighting parameter for external and internal forces.
        c1, c2 (float): Constants for the curvature term.
        c_g (float): Constant for the edge-stopping function g(h).
        T (int): Number of iterations for the evolution.
        timestep (float): Time step for the evolution.
        dirac_ratio (float): Scaling factor for the Dirac function.
        dt_max (float): Maximum value for Dirac normalization.
        epsilon (float): Threshold for boundary updates.

    Returns:
        torch.Tensor: Updated level set function.
    """
    EPSILON = 1e-10  # Small constant for numerical stability
    
    for k in range(T):
        # Apply Neumann boundary condition
        phi = NeumannBoundCond(phi)

        # Compute ∇phi and its magnitude
        phi_y, phi_x = gradient_sobel(phi, split=True)
        grad_phi_mag = torch.sqrt(phi_x**2 + phi_y**2 + EPSILON)

        # Normalize gradient to get N
        Nx = phi_x / grad_phi_mag
        Ny = phi_y / grad_phi_mag

        # Curvature k = div(N)
        curvature = div(Nx, Ny)

        # Compute ∇I and |∇I|
        I_y, I_x = gradient_sobel(I, split=True)
        grad_I_mag = torch.sqrt(I_x**2 + I_y**2 + EPSILON)

        # Edge-stopping function g(|∇I|)
        g_grad_I = g(grad_I_mag, c=c_g)

        # ∇g(|∇I|) ⋅ N
        g_grad_x, g_grad_y = gradient_sobel(g_grad_I, split=True)
        g_grad_dot_N = g_grad_x * Nx + g_grad_y * Ny

        # Dirac function
        diracPhi = Dirac(phi, dirac_ratio, dt_max=dt_max)

        # Forces
        common_term = g_grad_I * grad_phi_mag
        external_force = alpha * P * grad_phi_mag
        internal_force = (1 - alpha) * (c1 + c2 * curvature) * common_term
        correction_term = -(1 - alpha) * g_grad_dot_N * grad_phi_mag

        # Update with mask
        phi = phi + timestep * diracPhi * (external_force + internal_force + correction_term)
        
        # Regularization
    return phi
