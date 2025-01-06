import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt

from dataloader.drive_loader import unpad_to_original

from loss import *

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
    total_cl_dice= torch.tensor(0.0, device=device)
    
    if args.save_mask or epoch == args.max_epoch - 1:
        # ディレクトリ生成
        out_dir= os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].to(device)
            masks = sample['transformed_mask'].to(device)
            
            preds, s_long, s_short, v_long = model(images)
            soft_skeleton_pred = soft_skeleton(preds)
            soft_skeleton_gt = soft_skeleton(masks)
        
            # 個別の損失を計算
            loss_main = Loss()(preds, masks, soft_skeleton_pred, soft_skeleton_gt, alpha=args.alpha)
            loss_cosine = CosineLoss()(v_long, sample['vessel_directions'].to(device))
            loss_anisotropic = AnisotropicLoss()(s_long, s_short, masks) 
        
            loss = args.lambda_main * loss_main + args.lambda_cosine * loss_cosine + args.lambda_anisotropic * loss_anisotropic
        
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            mask_size = masks.shape[2:]
            preds_resized = F.interpolate(preds, size=mask_size, mode='bilinear')
            if args.rank == 0 and (args.save_mask or epoch == args.max_epoch - 1):
                save_main_out_image(preds_resized, os.path.join(out_dir, f"{i}.png"))
                save_main_out_image(preds_resized > args.threshold, os.path.join(out_dir, f"{i}_binary.png"), cmap="gray")
                save_main_out_image(s_long, os.path.join(out_dir, f"{i}_lambda_long.png"))
                save_main_out_image(s_short, os.path.join(out_dir, f"{i}_lambda_short.png"))
                visualize_vector_field_with_image(v_long, images, os.path.join(out_dir, f"{i}_v_long.png"), step=5, scale=0.1)
                
            # モデル出力が確率場解釈なので閾値で直接バイナリ化
            masks_pred = (preds_resized > args.threshold).float()
            # 元画像サイズにする
            masks_pred = unpad_to_original(masks_pred, sample["padding"])
            masks = unpad_to_original(masks, sample["padding"])
            
            # tp, tn, fp, fn
            tp = torch.sum((masks_pred == 1) & (masks == 1)).item()
            tn = torch.sum((masks_pred == 0) & (masks == 0)).item()
            fp = torch.sum((masks_pred == 1) & (masks == 0)).item()
            fn = torch.sum((masks_pred == 0) & (masks == 1)).item()
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            # dice
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            total_dice += dice
            
            # cl dice
            soft_skeleton_pred = soft_skeleton(masks_pred)
            soft_skeleton_gt = soft_skeleton(masks)
            tprec = (torch.sum(soft_skeleton_pred * masks) + 1) / (torch.sum(soft_skeleton_pred) + 1)
            tsens = (torch.sum(soft_skeleton_gt * masks_pred) + 1) / (torch.sum(soft_skeleton_gt) + 1)
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
    

def visualize_vector_field_with_image(vector_field, image, file_path, step=10, scale=0.1, cmap="gray"):
    """
    元画像を背景にベクトル場を可視化
    Args:
        vector_field (torch.Tensor): ベクトル場 (1, 2, H, W)
        image (torch.Tensor): 元画像 (1, 1, H, W) または (1, H, W)
        step (int): サンプリング間隔（矢印の密度を調整）
        scale (float): 矢印のスケール（値を小さくすると矢印が大きくなる）
        cmap (str): 背景画像のカラーマップ（デフォルトは 'gray'）
        file_path (str): 保存先のファイルパス
    """
    # ベクトル場の形状を確認
    assert vector_field.shape[0] == 1 and vector_field.shape[1] == 2, \
        f"Expected shape (1, 2, H, W), but got {vector_field.shape}"
    assert image.dim() in {3, 4}, \
        f"Expected shape (1, 1, H, W) or (1, H, W), but got {image.shape}"

    # バッチ次元を除外
    vector_field = vector_field.squeeze(0)  # (2, H, W)
    image = image.squeeze(0).squeeze(0) if image.dim() == 4 else image.squeeze(0)  # (H, W)

    # x成分とy成分を取得
    u = vector_field[0].detach().cpu().numpy()  # (H, W)
    v = vector_field[1].detach().cpu().numpy()  # (H, W)

    # サンプリング
    H, W = u.shape
    x, y = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    u_sampled = u[::step, ::step]
    v_sampled = v[::step, ::step]

    # 元画像を背景にベクトル場をプロット
    plt.figure(figsize=(8, 8))
    plt.imshow(image.detach().cpu().numpy(), cmap=cmap, extent=[0, W, H, 0])
    plt.quiver(x, y, u_sampled, v_sampled, angles='xy', scale_units='xy', scale=scale, color='blue')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.gca().invert_yaxis()  # 画像のy軸方向と一致させる
    plt.savefig(file_path)
    plt.close()


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
    inverted_input = 1 - input # 入力を反転
    return 1 - F.max_pool2d(inverted_input, kernel_size, stride, padding) # 最大プーリングを適用して再度反転


def soft_skeleton(mask, k=30):
    """
    ソフトスケルトン変換
    Args:
        mask (torch.Tensor): マスク画像 (N, 1, H, W)
        k (int): 最大管幅
    Returns:
        torch.Tensor: ソフトスケルトン画像 (N, 1, H, W)
    """
    # Initialize I' as maxpool(minpool(mask))
    I_prime = F.max_pool2d(minpool(mask, kernel_size=3, stride=1, padding=1), kernel_size=3, stride=1, padding=1)
    # Initialize S as ReLU(I - I')
    S = F.relu(mask - I_prime)

    # Iterative refinement of the skeleton
    for _ in range(k):
        # Update I
        mask = minpool(mask, kernel_size=3, stride=1, padding=1)
        # Update I'
        I_prime = F.max_pool2d(minpool(mask, kernel_size=3, stride=1, padding=1), kernel_size=3, stride=1, padding=1)
        # Update S
        S = S + (1 - S) * F.relu(mask - I_prime)

    return S