import torch
import torch.nn.functional as F

class BalancedBinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(BalancedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, preds, targets):
        # 正例と負例の数を計算
        num_positive = torch.sum(targets)
        num_negative = targets.numel() - num_positive
        num_total = num_positive + num_negative

        # 正例と負例の重みを計算
        pos_weight = num_negative / (num_total + 1e-6)
        neg_weight = num_positive / (num_total + 1e-6)

        # 正例と負例の損失を計算
        positive_loss = -pos_weight * targets * torch.log(preds + 1e-6)
        negative_loss = -neg_weight * (1 - targets) * torch.log(1 - preds + 1e-6)

        # 合計損失を計算
        loss = positive_loss + negative_loss

        # 平均化して返す
        return loss.mean()


class BCELoss(torch.nn.Module):
    def forward(self, preds, targets):
        return F.binary_cross_entropy(preds, targets)


class DiceLoss(torch.nn.Module):
    def forward(self, preds, targets):
        smooth = 1e-6
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
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
        smooth = 1e-6
        tp = (preds * targets).sum()
        fn = ((1 - preds) * targets).sum()
        fp = (preds * (1 - targets)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fn + self.beta * fp + smooth)
        return 1 - tversky