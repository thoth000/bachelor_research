import torch
import torch.nn.functional as F
from pde.eikonal import Heaviside

def class_balanced_bce_loss(outputs, labels, size_average=False, batch_average=True):
    """
    クラスバランスを考慮した二値交差エントロピー損失 (複数クラス対応)
    :param outputs: モデルの予測出力 (phi_T) [batch_size, num_classes, H, W]
    :param labels: 正解ラベル (gts) [batch_size, num_classes, H, W]
    :param size_average: 出力全体の平均を取るかどうか
    :param batch_average: バッチ全体で平均を取るかどうか
    :return: バランスを考慮した損失
    """
    assert outputs.size() == labels.size(), f"Mismatch in size between outputs {outputs.size()} and labels {labels.size()}"

    # バッチサイズとクラス数を取得
    batch_size, num_classes, h, w = outputs.size()

    final_loss = 0.0
    for class_id in range(num_classes):
        class_outputs = outputs[:, class_id, :, :]
        class_labels = labels[:, class_id, :, :]

        num_labels_pos = torch.sum(class_labels)
        num_labels_neg = torch.sum(1.0 - class_labels)
        num_total = num_labels_pos + num_labels_neg

        loss_val = -(torch.mul(class_labels, torch.log(class_outputs + 1e-4)) + torch.mul((1.0 - class_labels), torch.log(1.0 - class_outputs + 1e-4)))

        loss_pos = torch.sum(torch.mul(class_labels, loss_val))
        loss_neg = torch.sum(torch.mul(1.0 - class_labels, loss_val))

        class_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg
        final_loss += class_loss

    # サイズまたはバッチ全体で平均化
    if size_average:
        final_loss /= torch.numel(labels)
    elif batch_average:
        final_loss /= batch_size

    return final_loss


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

    # バッチサイズとクラス数を取得
    batch_size, num_classes, h, w = output.size()

    # 各クラスごとの損失を計算
    final_loss = 0.0
    for class_id in range(num_classes):
        output_class = output[:, class_id, :, :]
        sdt_class = sdt[:, class_id, :, :]
        mse = lambda x: torch.mean(torch.pow(x, 2))
        sdt_loss = mse(sdt_class - output_class)
        final_loss += sdt_loss

    # チャネル方向で損失を平均化
    final_loss /= num_classes

    return alpha * final_loss


def vector_field_loss(vf_pred, vf_gt, sdt=None, _print=False):
    """
    ベクトル場の損失を計算する関数 (複数クラス対応)
    vf_pred: 予測ベクトル場 (batch_size, 2, num_classes, H, W)
    vf_gt: グラウンドトゥルースベクトル場 (batch_size, 2, num_classes, H, W)
    """
    # バッチサイズとクラス数を取得
    batch_size, num_classes, h, w = vf_pred.size(0), vf_pred.size(2), vf_pred.size(3), vf_pred.size(4)

    final_loss = 0.0
    for class_id in range(num_classes):
        vf_pred_class = vf_pred[:, :, class_id, :, :]
        vf_gt_class = vf_gt[:, :, class_id, :, :]

        # ベクトル場の正規化 (xとyの2チャネルを一度に正規化)
        vf_pred_class = F.normalize(vf_pred_class, p=2, dim=1)
        vf_gt_class = F.normalize(vf_gt_class, p=2, dim=1)

        # コサイン類似度を用いた角度誤差
        cos_dist = torch.sum(vf_pred_class * vf_gt_class, dim=1)  # 内積を計算
        angle_error = torch.acos(cos_dist.clamp(-1 + 1e-4, 1 - 1e-4))  # コサイン類似度の逆関数

        # 角度誤差の二乗平均を損失として計算
        angle_loss = torch.mean(angle_error ** 2)
        final_loss += angle_loss

    # チャネル方向のロスを平均化
    final_loss /= num_classes

    if _print:
        print(f'[vf_loss] Vector Field Loss: {final_loss.item()}')

    return final_loss


def LSE_output_loss(phi_T, gts, sdts, epsilon=-1, dt_max=30):
    """
    T ステップ後のレベルセット (phi_T) とグラウンドトゥルースの損失
    :param phi_T: モデルの予測レベルセット [batch_size, num_classes, H, W]
    :param gts: グラウンドトゥルースのラベル [batch_size, num_classes, H, W]
    :param sdts: 符号付距離変換 [batch_size, num_classes, H, W]
    :param epsilon: Heavisideの平滑化パラメータ
    :param dt_max: 時間ステップの最大値
    :return: LSEの出力に基づいた損失
    """
    # ピクセルレベルの損失 (class_balanced_bce_lossを使用)
    pixel_loss = class_balanced_bce_loss(Heaviside(phi_T, epsilon=epsilon), gts, size_average=True)
    
    # 全体の損失 (boundary lossを使わないケース)
    loss = 100 * pixel_loss
    
    return loss
