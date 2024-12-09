import torch
import torch.nn.functional as F

from models.lse import *

def dice_loss(P, Y):
    """
    Compute the Dice loss.

    Args:
        P (torch.Tensor): Predicted probabilities, shape [batch, 1, H, W].
        Y (torch.Tensor): Ground truth labels, shape [batch, 1, H, W].

    Returns:
        torch.Tensor: Dice loss.
    """
    # Flatten tensors to compute intersections and unions
    P_flat = P.view(P.shape[0], -1)
    Y_flat = Y.view(Y.shape[0], -1)
    
    # Compute Dice loss
    intersection = (P_flat * Y_flat).sum(dim=1)
    dice = 1 - (2 * intersection) / (P_flat.sum(dim=1) + Y_flat.sum(dim=1) + 1e-8)
    return dice.mean()

def cross_entropy_loss(P, Y):
    """
    Compute the Cross-Entropy loss.

    Args:
        P (torch.Tensor): Predicted probabilities, shape [batch, 1, H, W].
        Y (torch.Tensor): Ground truth labels, shape [batch, 1, H, W].

    Returns:
        torch.Tensor: Cross-Entropy loss.
    """
    return -torch.mean(Y * torch.log(P + 1e-8) + (1 - Y) * torch.log(1 - P + 1e-8))

def L_LS_self(z, y, alpha=1.0, beta=1.0):
    """
    Compute the level set loss function L_LSself.

    Args:
        z (torch.Tensor): Network output before softmax, shape [batch, 2, H, W].
        y (torch.Tensor): Softmax output, shape [batch, 2, H, W].
        alpha (float): Weight for the first term.
        beta (float): Weight for the second term.

    Returns:
        torch.Tensor: Computed loss value.
    """
    # Split z and y into components
    z1 = z[:, 0:1, :, :]  # [batch, 1, H, W]
    z2 = z[:, 1:2, :, :]  # [batch, 1, H, W]
    y1 = y[:, 0:1, :, :]  # [batch, 1, H, W]
    y2 = y[:, 1:2, :, :]  # [batch, 1, H, W]

    # Compute c1 and c2
    c1 = (torch.sum(z1 * y1, dim=(2, 3)) / torch.sum(y1, dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
    c2 = (torch.sum(z2 * y2, dim=(2, 3)) / torch.sum(y2, dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)

    # Compute the loss terms
    loss_1 = torch.sum(((z1 - c1) ** 2) * y1, dim=(2, 3))
    loss_2 = torch.sum(((z2 - c2) ** 2) * y2, dim=(2, 3))

    # Total loss with weights alpha and beta
    loss = alpha * torch.mean(loss_1) + beta * torch.mean(loss_2)
    return loss


def L_LS(z, y, alpha=1.0, beta=1.0):
    """
    Compute the level set loss function L_LS.

    Args:
        z (torch.Tensor): Network output before softmax, shape [batch, 2, H, W].
        y (torch.Tensor): Ground truth, shape [batch, 1, H, W].
        alpha (float): Weight for the first term.
        beta (float): Weight for the second term.

    Returns:
        torch.Tensor: Computed loss value.
    """
    # Split z into components
    z1 = z[:, 0:1, :, :]  # [batch, 1, H, W]
    z2 = z[:, 1:2, :, :]  # [batch, 1, H, W]

    # Compute c1 and c2
    c1 = (torch.sum(z1 * y, dim=(2, 3)) / torch.sum(y, dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
    c2 = (torch.sum(z2 * (1 - y), dim=(2, 3)) / torch.sum(1 - y, dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)

    # Compute the loss terms
    
    loss_1 = torch.sum(((z1 - c1) ** 2) * y, dim=(2, 3))
    loss_2 = torch.sum(((z2 - c2) ** 2) * (1 - y), dim=(2, 3))


    # Total loss with weights alpha and beta
    loss = alpha * torch.mean(loss_1) + beta * torch.mean(loss_2)
    return loss


def selected_loss(args, y1, gt, z, y, mode="pre_loss"):
    if args.criterion == "ce":
        return cross_entropy_loss(y1, gt)
    elif args.criterion == "ce_dice":
        return cross_entropy_loss(y1, gt) + dice_loss(y1, gt)
    elif args.criterion == "ce_dice_ls":
        if mode == "pre_loss":
            return cross_entropy_loss(y1, gt) + dice_loss(y1, gt) + 1e-6 * L_LS_self(z, y)
        else:
            return cross_entropy_loss(y1, gt) + dice_loss(y1, gt) + 1e-6 * L_LS(z, gt)