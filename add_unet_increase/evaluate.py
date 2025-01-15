import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt

from dataloader.drive_loader import unpad_to_original

from loss import *

def minpool(input, kernel_size=3, stride=1, padding=1):
    """
    最小プーリング
    Args:
        input (torch.Tensor): 入力テンソル (N, C, H, W)
        kernel_size (int): カーネルサイズ
        stride (int): ストライド
        padding (int): パディング
    Returns:
        torch.Tensor: 出力テンソル (N, C, H, W)
    """
    return F.max_pool2d(input*-1, kernel_size, stride, padding)*-1 # 最大プーリングを適用して再度反転


def soft_skeleton(x, thresh_width=30):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def create_anisotropic_tensor_from_vector(vectors, lambda1=1.0, lambda2=0.0):
    """
    方向ベクトルに基づく異方性拡散テンソルを生成。

    Parameters:
        vectors (torch.Tensor): 方向ベクトル（形状: [B, 2, H, W]）。
        lambda1 (float): 主方向の拡散強度。
        lambda2 (float): 直交方向の拡散強度。

    Returns:
        D (torch.Tensor): 異方性拡散テンソル（形状: [B, 2, 2, H, W]）。
    """
    B, _, H, W = vectors.shape

    # ベクトルを正規化
    v_x = vectors[:, 0].unsqueeze(1)  # [B, 1, H, W]
    v_y = vectors[:, 1].unsqueeze(1)  # [B, 1, H, W]
    # print("v_x.shape", v_x.shape, "v_y.shape", v_y.shape)  # [64, 1, 128, 128]

    # テンソル構築
    # print("v_x * v_x", (v_x * v_x).shape)
    # print("stack.shape", torch.stack([v_x * v_x, v_x * v_y], dim=1).shape)  # [64, 1, 128, 128]
    
    vvT = torch.zeros(B, 2, 2, H, W, device=vectors.device, dtype=vectors.dtype)  # [B, 2, 2, H, W]
    vvT[:, 0, 0] = (v_x * v_x).squeeze(1)  # [B, H, W]
    vvT[:, 0, 1] = (v_x * v_y).squeeze(1)  # [B, H, W]
    vvT[:, 1, 0] = (v_y * v_x).squeeze(1)  # [B, H, W]
    vvT[:, 1, 1] = (v_y * v_y).squeeze(1)  # [B, H, W]

    # print("vvT.shape", vvT.shape)  # [64, 2, 2, 128, 128]

    I = torch.eye(2, device=vectors.device).view(1, 2, 2, 1, 1)  # 単位行列
    D = lambda1 * vvT + lambda2 * (I - vvT)

    return D


def gradient(scalar_field):
    """
    スカラー場の勾配を計算。

    Parameters:
        scalar_field (torch.Tensor): スカラー場（形状: [B, 1, H, W]）。

    Returns:
        grad_x (torch.Tensor): x方向の勾配（形状: [B, 1, H, W]）。
        grad_y (torch.Tensor): y方向の勾配（形状: [B, 1, H, W]）。
    """
    # 有限差分係数（精度8次）
    coeff = torch.tensor([-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280],
                         dtype=scalar_field.dtype, device=scalar_field.device)

    # x方向の勾配計算
    x_pad = F.pad(scalar_field, (4, 4, 0, 0), mode='replicate')
    grad_x = sum(coeff[i] * x_pad[..., i:i+scalar_field.size(-1)] for i in range(9))

    # y方向の勾配計算
    y_pad = F.pad(scalar_field, (0, 0, 4, 4), mode='replicate')
    grad_y = sum(coeff[i] * y_pad[..., i:i+scalar_field.size(-2), :] for i in range(9))

    return grad_x, grad_y


def divergence(grad_x, grad_y):
    """
    ベクトル場の発散を計算。

    Parameters:
        grad_x (torch.Tensor): x方向のベクトル場（形状: [B, 1, H, W]）。
        grad_y (torch.Tensor): y方向のベクトル場（形状: [B, 1, H, W]）。

    Returns:
        divergence (torch.Tensor): 発散（形状: [B, 1, H, W]）。
    """
    # 有限差分係数（精度8次）
    coeff = torch.tensor([-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280],
                         dtype=grad_x.dtype, device=grad_x.device)

    # x方向の発散計算
    dx_pad = F.pad(grad_x, (4, 4, 0, 0), mode='replicate')
    div_x = sum(coeff[i] * dx_pad[..., i:i+grad_x.size(-1)] for i in range(9))

    # y方向の発散計算
    dy_pad = F.pad(grad_y, (0, 0, 4, 4), mode='replicate')
    div_y = sum(coeff[i] * dy_pad[..., i:i+grad_y.size(-2), :] for i in range(9))

    return div_x + div_y


def anisotropic_diffusion(preds, diffusion_tensor, num_iterations=10, gamma=0.1):
    """
    異方性拡散プロセスを実行。

    Parameters:
        preds (torch.Tensor): スカラー場（形状: [B, 1, H, W]）。
        diffusion_tensor (torch.Tensor): 異方性拡散テンソル（形状: [B, 2, 2, H, W]）。
        num_iterations (int): 拡散の反復回数。
        gamma (float): 拡散の強さ。

    Returns:
        preds (torch.Tensor): 拡散後のスカラー場（形状: [B, 1, H, W]）。
    """
    B, _, H, W = preds.shape

    for _ in range(num_iterations):
        # 勾配計算
        grad_x, grad_y = gradient(preds)  # [B, 1, H, W]
        grad = torch.cat([grad_x, grad_y], dim=1)  # [B, 2, H, W]
        # print("grad.shape", grad.shape)  # [64, 2, 128, 128]

        # print("diffusion_tensor.shape", diffusion_tensor.shape)  # [64, 2, 2, 128, 128]

        # 勾配を異方性テンソルで変換
        transformed_grad = torch.einsum('bijhw,bjhw->bihw', diffusion_tensor, grad)  # [B, 2, H, W]

        # ダイバージェンス計算
        div_x, div_y = transformed_grad[:, 0], transformed_grad[:, 1]
        div = divergence(div_x.unsqueeze(1), div_y.unsqueeze(1))  # [B, 1, H, W]

        div = F.relu(div)  # マイナス値を0にクリップ

        # 拡散更新
        preds = preds + gamma * div

        preds = preds.clamp(0, 1)  # 0から1の範囲にクリップ

    return preds


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

        # soft cl dice loss (バッチ次元ごとに計算)
        tprec = (torch.sum(soft_skeleton_pred * masks_gt, dim=1) + 1) / \
                (torch.sum(soft_skeleton_pred, dim=1) + 1)
        tsens = (torch.sum(soft_skeleton_gt * preds, dim=1) + 1) / \
                (torch.sum(soft_skeleton_gt, dim=1) + 1)
        cl_dice_loss = 1 - (2 * tprec * tsens) / (tprec + tsens)

        # バッチ次元ごとに損失を組み合わせ
        loss = (1 - alpha) * dice_loss + alpha * cl_dice_loss

        # バッチ全体の損失を平均化
        return loss.mean()


# loss, accuracy, IoU, Dice
def evaluate(model, dataloader, criterion, epoch, args, device):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    total_dice = torch.tensor(0.0, device=device)
    total_cl_dice = torch.tensor(0.0, device=device)
    
    if args.save_mask or epoch == args.max_epoch - 1:
        # ディレクトリ生成
        out_dir= os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].to(device)
            masks_gt = sample['transformed_mask'].to(device)
            
            preds_origin, vec = model(images)
            preds_origin = torch.sigmoid(preds_origin)
            vec = F.normalize(vec, p=2, dim=1)
            # 異方性拡散を適用
            preds = anisotropic_diffusion(preds_origin.clone(), create_anisotropic_tensor_from_vector(vec), num_iterations=args.num_iterations, gamma=args.gamma)
            
            soft_skeleton_pred = soft_skeleton(preds.clone())
            soft_skeleton_gt = soft_skeleton(masks_gt.clone())
            
            loss = Loss()(preds, masks_gt, soft_skeleton_pred, soft_skeleton_gt, alpha=args.alpha)
            # loss用ではなく，評価指標用
            
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            
            preds_unpad = unpad_to_original(preds.clone(), sample["padding"])
            masks_unpad = unpad_to_original(masks.clone(), sample["padding"])
            masks_pred_unpad = (preds_unpad > args.threshold).float()
            soft_skeleton_pred = soft_skeleton(masks_pred_unpad.clone())
            soft_skeleton_gt = soft_skeleton(masks_unpad.clone())
            
            # save image
            if args.rank == 0 and args.save_mask and i == 0:
                # masks
                save_binary_image(masks_pred_unpad, os.path.join(out_dir, f"mask_pred_{i}.png"))
                save_binary_image(masks_unpad, os.path.join(out_dir, f"mask_gt_{i}.png"))
                # soft skeleton
                save_binary_image(soft_skeleton_pred, os.path.join(out_dir, f"soft_skeleton_pred_{i}.png"))
                save_binary_image(soft_skeleton_gt, os.path.join(out_dir, f"soft_skeleton_gt_{i}.png"))
                # preds
                save_main_out_image(preds, os.path.join(out_dir, f"pred_anisotropic{i}.png"))
                save_main_out_image(preds_origin, os.path.join(out_dir, f"pred_origin_{i}.png"))
                save_main_out_image(preds - preds_origin, os.path.join(out_dir, f"diff_{i}.png"))
                # directions
                save_main_out_image(vec[:, 0:1], os.path.join(out_dir, f"vec_x_{i}.png"))
                save_main_out_image(vec[:, 1:2], os.path.join(out_dir, f"vec_y_{i}.png"))
                # masked directions
                save_masked_output_with_imsave(vec[:, 0:1], masks_gt, os.path.join(out_dir, f"masked_vec_x_{i}.png"))
                save_masked_output_with_imsave(vec[:, 1:2], masks_gt, os.path.join(out_dir, f"masked_vec_y_{i}.png"))
                # masked absolute directions
                save_masked_output_with_imsave(torch.abs(vec[:, 0:1]), masks_gt, os.path.join(out_dir, f"masked_abs_vec_x_{i}.png"))
                save_masked_output_with_imsave(torch.abs(vec[:, 1:2]), masks_gt, os.path.join(out_dir, f"masked_abs_vec_y_{i}.png"))

                
            # tp, tn, fp, fn
            tp = torch.sum((masks_pred_unpad == 1) & (masks_unpad == 1)).item()
            tn = torch.sum((masks_pred_unpad == 0) & (masks_unpad == 0)).item()
            fp = torch.sum((masks_pred_unpad == 1) & (masks_unpad == 0)).item()
            fn = torch.sum((masks_pred_unpad == 0) & (masks_unpad == 1)).item()
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            # Dice
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            total_dice += dice
            
            # cl dice
            tprec = (torch.sum(soft_skeleton_pred * masks_unpad) + 1) / (torch.sum(soft_skeleton_pred) + 1)
            tsens = (torch.sum(soft_skeleton_gt * masks_pred_unpad) + 1) / (torch.sum(soft_skeleton_gt) + 1)
            cl_dice = (2 * tprec * tsens) / (tprec + tsens)
            total_cl_dice += cl_dice
    
    # 集計
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_dice, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_cl_dice, op=dist.ReduceOp.SUM)
    
    avg_loss = running_loss / total_samples
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn)
    spe = total_tn / (total_tn + total_fp)
    iou = total_tp / (total_tp + total_fp + total_fn)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    # dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) # F1
    dice = total_dice / total_samples
    cl_dice = total_cl_dice / total_samples
    
    return avg_loss.item(), acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item(), cl_dice.item()


def save_main_out_image(output_tensor, filepath, cmap="viridis"):
    """
    output_tensorをカラーマップで画像として保存する関数
    :param output_tensor: 保存するテンソル
    :param filepath: 画像の保存先のパス
    :param cmap: 使用するカラーマップ（デフォルトは 'viridis'）
    """
    # output_tensorをnumpy配列に変換
    output_numpy = output_tensor.squeeze().cpu().detach().numpy()

    # カラーマップを使って画像として保存
    plt.imshow(output_numpy, cmap=cmap)
    plt.colorbar()
    plt.savefig(filepath)  # ファイルパスに画像を保存
    plt.close()  # メモリを解放するためにプロットを閉じる

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def save_masked_output(output_tensor, mask_tensor, filepath, cmap="viridis", vmin=-1, vmax=1):
    """
    真値マスクで指定された部分だけをカラーマップで可視化して保存する関数。
    
    Args:
        output_tensor (torch.Tensor): 可視化したい出力テンソル（shape: [1, 1, H, W]）。
        mask_tensor (torch.Tensor): 01バイナリの真値マスクテンソル [1, 1, H, W]。
        filepath (str): 保存するファイルパス。
        cmap (str): 使用するカラーマップ（デフォルトは 'viridis'）。
    """
    # 出力テンソルとマスクテンソルを NumPy 配列に変換（[H, W] に変換）
    
    # print("torch", torch.sum(mask_tensor))
    
    output_numpy = output_tensor[0, 0].cpu().detach().numpy()
    mask_numpy = mask_tensor[0, 0].cpu().detach().numpy()

    # print("np", np.sum(mask_numpy))
    # print("np.float", np.sum(mask_numpy > 0.5))

    # マスクを適用して出力テンソルをフィルタリング
    masked_output = np.where(mask_numpy > 0.5, output_numpy, np.nan)

    # カラーマップを使って画像として保存
    plt.imshow(masked_output, cmap=cmap, vmin=vmin, vmax=vmax)  # vmin/vmaxで値域を明示
    plt.colorbar()
    plt.savefig(filepath)
    plt.close()


import matplotlib.pyplot as plt
import torch
import numpy as np

import matplotlib.pyplot as plt
import torch
import numpy as np

def save_masked_output_with_imsave(output_tensor, mask_tensor, filepath, normalize=True, cmap="viridis", vmin=-1, vmax=1):
    """
    真値マスクで指定された部分だけをカラーマップで可視化して保存する関数（plt.imsaveを使用）。
    
    Args:
        output_tensor (torch.Tensor): 可視化したい出力テンソル（shape: [1, 1, H, W]）。
        mask_tensor (torch.Tensor): 01バイナリの真値マスクテンソル [1, 1, H, W]。
        filepath (str): 保存するファイルパス。
        normalize (bool): 出力テンソルを [0, 1] に正規化するかどうか。
        cmap (str): 使用するカラーマップ（デフォルトは 'viridis'）。
    """
    # 出力テンソルとマスクテンソルを NumPy 配列に変換（[H, W] に変換）
    output_numpy = output_tensor[0, 0].cpu().detach().numpy()
    mask_numpy = mask_tensor[0, 0].cpu().detach().numpy()

    # マスクを適用して出力テンソルをフィルタリング
    masked_output = np.where(mask_numpy > 0.5, output_numpy, np.nan)

    # `np.nan` をカラーマップの白色に置き換え
    masked_output = np.ma.masked_invalid(masked_output)  # NaNをマスク
    cmap_instance = plt.cm.get_cmap(cmap)  # カラーマップを取得
    cmap_instance.set_bad(color='white')  # マスクされた部分を白色に設定

    # `plt.imsave` を使用して保存
    plt.imsave(filepath, masked_output, cmap=cmap_instance, vmin=vmin, vmax=vmax)




import PIL.Image as Image
def save_binary_image(output_tensor, filepath):
    """
    output_tensorを2値画像として保存する関数
    :param output_tensor: 保存するテンソル
    :param filepath: 画像の保存先のパス
    """
    # output_tensorをnumpy配列に変換
    output_numpy = output_tensor.squeeze().cpu().detach().numpy()
    output_numpy = (output_numpy * 255).astype(np.uint8)
    # PIL Imageに変換
    output_image = Image.fromarray(output_numpy, mode='L')
    # 画像を保存
    output_image.save(filepath)

