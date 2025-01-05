import torch
import torch.nn.functional as F

class BalancedBinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(BalancedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        # 正例と負例の数を計算
        num_positive = torch.sum(targets)
        num_negative = targets.numel() - num_positive

        # pos_weight の計算
        pos_weight = num_negative / (num_positive + 1e-6)  # 正例の重要度を調整

        # Binary Cross Entropy with logits and pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight,  # 正例の重みを指定
            reduction='mean'
        )
        
        return loss

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

class SoftClDiceLoss(torch.nn.Module):
    def forward(self, Vp, Vl, Sp, Sl):
        N = Vp.size(0)
        Vp = Vp.view(N, -1)
        Vl = Vl.view(N, -1)
        Sp = Sp.view(N, -1)
        Sl = Sl.view(N, -1)
        T_prec = ((Sp * Vl).sum(dim=1) + 1e-6) / ((Sp).sum(dim=1) + 1e-6)
        T_sens = ((Sl * Vp).sum(dim=1) + 1e-6) / ((Sl).sum(dim=1) + 1e-6)
        clDice = 2 * (T_prec * T_sens) / (T_prec + T_sens)
        return (1 - clDice).mean()


class GeneralizedDiceLoss(torch.nn.Module):
    def forward(self, preds, targets):
        smooth = 1e-6
        N = targets.size(0)
        preds = preds.view(N, -1)
        targets = targets.view(N, -1)
        intersection = (preds * targets).sum(dim=1)
        cardinality = (preds**2).sum(dim=1) + (targets**2).sum(dim=1)
        dice = 1 - 2 * intersection / (cardinality + smooth)
        return dice.mean()
