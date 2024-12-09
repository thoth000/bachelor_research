import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def forward(self, preds, targets):
        smooth = 1e-6
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        smooth = 1e-6
        tp = (preds * targets).sum()
        fn = ((1 - preds) * targets).sum()
        fp = (preds * (1 - targets)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fn + self.beta * fp + smooth)
        return 1 - tversky