import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt

from dataloader.drive_loader import unpad_to_original

# loss, accuracy, IoU, Dice
def evaluate(model, dataloader, criterion, epoch, args, device):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    if args.save_mask or epoch == args.max_epoch - 1:
        # ディレクトリ生成
        out_dir= os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].to(device)
            targets = sample['transformed_mask'].to(device)
            
            main_out, s_long, s_short, v_long = model(images)
            loss = criterion(main_out, targets)
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            mask_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=mask_size, mode='bilinear')
            if args.rank == 0 and (args.save_mask or epoch == args.max_epoch - 1):
                save_main_out_image(main_out_resized, os.path.join(out_dir, f"{i}.png"))
                save_main_out_image(main_out_resized > args.threshold, os.path.join(out_dir, f"{i}_binary.png"), cmap="gray")
                save_main_out_image(s_long, os.path.join(out_dir, f"{i}_lambda_long.png"))
                save_main_out_image(s_short, os.path.join(out_dir, f"{i}_lambda_short.png"))
                visualize_vector_field_with_image(v_long, images, os.path.join(out_dir, f"{i}_v_long.png"), step=10, scale=0.1)
                

            # モデル出力が確率場解釈なので閾値で直接バイナリ化
            preds = (main_out_resized > args.threshold).float()
            # 元画像サイズにする
            preds = unpad_to_original(preds, sample["padding"])
            masks = unpad_to_original(masks, sample["padding"])
            
            # tp, tn, fp, fn
            total_tp += torch.sum((preds == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds == 0) & (masks == 1)).item()
    
    # 集計
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    
    avg_loss = running_loss / total_samples
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn)
    spe = total_tn / (total_tn + total_fp)
    iou = total_tp / (total_tp + total_fp + total_fn)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) # F1
    
    return avg_loss.item(), acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item()


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
