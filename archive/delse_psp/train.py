import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pde.eikonal import levelset_evolution, gradient_sobel
from myconfig import config
from layers.lse import LSE_output_loss, vector_field_loss, level_map_loss
from tqdm import tqdm


def compute_grad_norm(model, norm_type=2):
    """
    モデルの勾配ノルムを計算する関数
    :param model: PyTorchのモデル
    :param norm_type: ノルムの種類 (デフォルトはL2ノルム)
    :return: 総勾配ノルム
    """
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train_one_epoch(model, dataloader, optimizer, device):
    torch.cuda.empty_cache()
    model.train()
    running_loss = 0.0

    for sample in dataloader:
        # 入力画像、符号付距離(sdt)、グラウンドトゥルース(gts)を取得
        images = sample['transformed_image'].to(device)
        sdts = sample['sdt'].to(device)  # 符号付距離
        gts = sample['transformed_mask'].to(device)  # グラウンドトゥルース
        dts = sample['dt'].to(device)  # 距離変換

        # モデルの出力 (phi_0, energy, g)
        phi_0, energy, g = model(images)

        # 出力のサイズを入力画像に揃える
        target_size = images.size()[2:]  # (H, W)サイズを取得
        phi_0 = F.interpolate(phi_0, size=target_size, mode='bilinear', align_corners=True)
        energy = F.interpolate(energy, size=target_size, mode='bilinear', align_corners=True)
        g = F.sigmoid(F.interpolate(g, size=target_size, mode='bilinear', align_corners=True))

        # ベクトル場gを x方向とy方向に分割 (batch_size, 2, num_classes, H, W)
        batch_size, channels, h, w = energy.size()
        energy = energy.view(batch_size, 2, channels // 2, h, w)  # xとy方向に分割

        # 初期レベルセットの損失を計算 (符号付距離に基づく)
        phi_0_loss = level_map_loss(phi_0, sdts, alpha=config['alpha'])

        # phi_Tのレベルセット進化 (レベルセット進化)
        rand_shift = 10 * np.random.rand() - 5
        phi_T = levelset_evolution(phi_0 + rand_shift, energy, g, T=config['T'], dt_max=config['dt_max'])

        # 距離変換 (dts) に基づいて勾配データ (vfs) を計算
        vfs = gradient_sobel(dts, split=False)

        # ベクトル場の損失
        vf_loss = vector_field_loss(energy, vfs, sdts)

        # phi_Tから0/1マスクを生成
        # predicted_mask = (phi_T <= 0).float() # [batch, channel, H, W]で各ピクセルに0,1
        # predicted_class = torch.argmax(predicted_mask, dim=1) # [batch, H, W]で各ピクセルにクラスID

        # Tステップ後のレベルセットとグラウンドトゥルースのロス
        phi_T_loss = LSE_output_loss(phi_T, gts, sdts, epsilon=config['epsilon'], dt_max=config['dt_max'])

        # 総損失
        # print(phi_0_loss.item(), phi_T_loss.item(), vf_loss.item())
        loss = phi_0_loss + phi_T_loss + vf_loss

        # 勾配計算
        loss.backward()
        
        # 勾配ノルムを計算して出力
        # grad_norm = compute_grad_norm(model, norm_type=2)
        # print(f"Gradient Norm: {grad_norm}") # 50000程度で抑える
        
        # 勾配クリッピング
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
        # 勾配ノルムを計算して出力
        # grad_norm = compute_grad_norm(model, norm_type=2)
        # print(f"Gradient Norm: {grad_norm}, Loss: {[phi_0_loss.item(), phi_T_loss.item(), vf_loss.item()]}")
        
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss
