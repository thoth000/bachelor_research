import torch
import torch.nn.functional as F

from models.lse import *

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

def P_loss(P_pred, P_gt):
    """
    P の損失関数
    
    :param P_pred: モデルの予測 (P) [batch_size, num_classes, H, W]
    :param P_gt: 正解データ (P) [batch_size, num_classes, H, W]
    """
    return torch.nn.MSELoss()(P_pred, P_gt)

