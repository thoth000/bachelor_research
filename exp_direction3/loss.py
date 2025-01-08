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


class CosineLoss(torch.nn.Module):
    def forward(self, preds, targets):
        preds = F.normalize(preds, p=2, dim=1)
        targets = F.normalize(targets, p=2, dim=1)
        
        dot_product = (preds * targets).sum(dim=1)
        loss = 1 - dot_product
        return loss.mean()


class AnisotropicLoss(torch.nn.Module):
    def forward(self, s_long, s_short, mask):
        return torch.sum(F.relu(s_short - s_long) * mask) / (mask.sum() + 1e-7)


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


class Loss(torch.nn.Module):
    def forward(self, preds, masks_gt, soft_skeleton_pred, soft_skeleton_gt, alpha=0.5):
        # バッチ次元を保持し、他の次元をフラット化
        batch_size = masks_gt.size(0)
        preds = preds.view(batch_size, -1)
        masks_gt = masks_gt.view(batch_size, -1)
        soft_skeleton_pred = soft_skeleton_pred.view(batch_size, -1)
        soft_skeleton_gt = soft_skeleton_gt.view(batch_size, -1)

        # soft dice loss (バッチ次元ごとに計算)
        dice_loss = 1 - (2 * torch.sum(masks_gt * preds, dim=1) + 1) / \
                    (torch.sum(masks_gt, dim=1) + torch.sum(preds, dim=1) + 1)

        # bce_loss = F.binary_cross_entropy(preds, masks_gt, reduction="none").sum(dim=1)

        # soft cl dice loss (バッチ次元ごとに計算)
        tprec = (torch.sum(soft_skeleton_pred * masks_gt, dim=1) + 1) / \
                (torch.sum(soft_skeleton_pred, dim=1) + 1)
        tsens = (torch.sum(soft_skeleton_gt * preds, dim=1) + 1) / \
                (torch.sum(soft_skeleton_gt, dim=1) + 1)
        cl_dice_loss = 1 - (2 * tprec * tsens) / (tprec + tsens)

        # バッチ次元ごとに損失を組み合わせ
        loss = (1 - alpha) * dice_loss + alpha * cl_dice_loss
        # loss = (1 - alpha) * bce_loss + alpha * cl_dice_loss

        # バッチ全体の損失を平均化
        return loss.mean()