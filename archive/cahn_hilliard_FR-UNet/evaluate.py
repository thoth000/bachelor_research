import os
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
            
            main_out = model(images)
            loss = criterion(main_out, targets)
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            mask_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=mask_size, mode='bilinear')
            if args.save_mask or epoch == args.max_epoch - 1:
                save_main_out_image(main_out_resized, os.path.join(out_dir, f"{i}.png"))
            
            preds = torch.sigmoid(main_out_resized)
            # クラスに不均衡があり、1が少ないので閾値を0.5より低めに設定する
            preds = (preds > args.threshold).float()
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
