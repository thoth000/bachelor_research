import torch
import torch.nn.functional as F

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


def selected_loss(args, y1, gt, z, y):
    if args.criterion == "ce":
        return cross_entropy_loss(y1, gt)
    elif args.criterion == "ce_dice":
        return cross_entropy_loss(y1, gt) + dice_loss(y1, gt)