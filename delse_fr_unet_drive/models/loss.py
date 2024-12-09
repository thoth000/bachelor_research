import torch
import torch.nn.functional as F

from models.lse import *

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

def class_balanced_bce_loss(outputs, labels, size_average=False, batch_average=True):
    assert(outputs.size() == labels.size())

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    loss_val = -(torch.mul(labels, torch.log(outputs)) + torch.mul((1.0 - labels), torch.log(1.0 - outputs)))

    loss_pos = torch.sum(torch.mul(labels, loss_val))
    loss_neg = torch.sum(torch.mul(1.0 - labels, loss_val))

    final_loss = (num_labels_neg / num_total) * loss_pos + (num_labels_pos / num_total) * loss_neg

    if size_average:
        final_loss /= torch.numel(labels)
    elif batch_average:
        final_loss /= labels.size()[0]
    
    return final_loss

def LSE_output_loss(phi_T, gts, epsilon=-1, dt_max=30):
    pixel_loss = class_balanced_bce_loss(Heaviside(phi_T, epsilon=epsilon), gts, size_average=True)
    # boundary_loss = bd_loss(phi_T, sdts, sigma=2.0/dt_max, dt_max=dt_max)
    loss = 100 * pixel_loss
    
    # print(f'[LSE_output_loss] Loss: {loss.item()}')
    return loss

def level_map_loss(output, sdt, alpha=1):
    """
    初期レベルセット (phi_0) と符号付距離 (sdt) の損失
    :param output: モデルの予測 (phi_0) [batch_size, num_classes, H, W]
    :param sdt: 符号付距離変換 (signed distance transform) [batch_size, num_classes, H, W]
    :param alpha: 重み係数
    :return: レベルセットマップの損失
    """
    # 入力のサイズチェック
    if output.size() != sdt.size():
        raise ValueError(f"Size mismatch: output {output.size()} and sdt {sdt.size()} must be the same.")

    # h, w = output.size(2), output.size(3)
    
    # 各クラスごとの損失を計算
    mse = lambda x: torch.mean(torch.pow(x, 2))
    sdt_loss = mse(output - sdt)

    # print(f'[level_map_loss] Level Map Loss: {alpha * sdt_loss.item()}')

    return alpha * sdt_loss

def vector_field_loss(vx, vy, vx_gt, vy_gt, _print=False):
    """
    ベクトル場の損失を計算する関数 (複数クラス対応)
    vx: 予測ベクトル場の x 成分 (batch_size, num_classes, H, W)
    vy: 予測ベクトル場の y 成分 (batch_size, num_classes, H, W)
    vx_gt: グラウンドトゥルースベクトル場の x 成分 (batch_size, num_classes, H, W)
    vy_gt: グラウンドトゥルースベクトル場の y 成分 (batch_size, num_classes, H, W)
    """
    # バッチサイズとクラス数を取得
    batch_size, num_classes, h, w = vx.size(0), vx.size(1), vx.size(2), vx.size(3)
    
    final_loss = 0.0
    for class_id in range(num_classes):
        # 各クラスごとのベクトル場成分を抽出
        vx_pred_class = vx[:, class_id, :, :]
        vy_pred_class = vy[:, class_id, :, :]
        vx_gt_class = vx_gt[:, class_id, :, :]
        vy_gt_class = vy_gt[:, class_id, :, :]
        
        # ベクトル場の正規化 (2次元ベクトルを一度に正規化)
        pred_norm = torch.sqrt(vx_pred_class**2 + vy_pred_class**2 + 1e-10)
        gt_norm = torch.sqrt(vx_gt_class**2 + vy_gt_class**2 + 1e-10)
        
        vx_pred_class = vx_pred_class / pred_norm
        vy_pred_class = vy_pred_class / pred_norm
        vx_gt_class = vx_gt_class / gt_norm * -1
        vy_gt_class = vy_gt_class / gt_norm * -1
        
        # コサイン類似度を用いた角度誤差
        cos_dist = vx_pred_class * vx_gt_class + vy_pred_class * vy_gt_class  # 内積を計算
        angle_error = torch.acos(cos_dist.clamp(-1 + 1e-6, 1 - 1e-6))  # コサイン類似度の逆関数で角度誤差を計算
        
        # 角度誤差の二乗平均を損失として計算
        angle_loss = torch.mean(angle_error ** 2)
        final_loss += angle_loss

    # チャネル方向のロスを平均化
    final_loss /= num_classes

    if _print:
        print(f'[vf_loss] Vector Field Loss: {final_loss.item()}')

    return final_loss

