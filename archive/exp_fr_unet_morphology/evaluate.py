import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt

from dataloader.drive_loader import unpad_to_original


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
            
            preds = model(images)
            preds = torch.sigmoid(preds)
            soft_skeleton_pred = soft_skeleton(preds)
            soft_skeleton_gt = soft_skeleton(masks_gt)
            
            loss = Loss()(preds, masks_gt, soft_skeleton_pred, soft_skeleton_gt, alpha=args.alpha)
            # loss用ではなく，評価指標用
            
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            
            preds = unpad_to_original(preds, sample["padding"])
            masks = unpad_to_original(masks, sample["padding"])
            masks_pred = (preds > args.threshold).float()
            soft_skeleton_pred = (soft_skeleton(preds) > args.threshold).float()
            soft_skeleton_gt = soft_skeleton(masks)
            
            # save image
            if args.rank == 0 and args.save_mask and i == 0:
                save_binary_image(masks_pred, os.path.join(out_dir, f"pred_{i}.png"))
                save_binary_image(masks, os.path.join(out_dir, f"gt_{i}.png"))
                save_binary_image(soft_skeleton_pred, os.path.join(out_dir, f"soft_skeleton_pred_{i}.png"))
                save_binary_image(soft_skeleton_gt, os.path.join(out_dir, f"soft_skeleton_gt_{i}.png"))
                
            # tp, tn, fp, fn
            tp = torch.sum((masks_pred == 1) & (masks == 1)).item()
            tn = torch.sum((masks_pred == 0) & (masks == 0)).item()
            fp = torch.sum((masks_pred == 1) & (masks == 0)).item()
            fn = torch.sum((masks_pred == 0) & (masks == 1)).item()
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            # Dice
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            total_dice += dice
            
            # cl dice
            tprec = (torch.sum(soft_skeleton_pred * masks_gt) + 1) / (torch.sum(soft_skeleton_pred) + 1)
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