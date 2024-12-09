import os
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

# loss, accuracy, IoU, Dice
def evaluate(model, dataloader, criterion, epoch, args):
    model.eval()
    running_loss = 0.0
    # accuracy
    total_correct = 0
    total_numel = 1e-6
    # IoU
    total_intersection = 0
    total_union = 1e-6
    # Dice
    total_joint = 1e-6
    
    # ディレクトリ生成
    out_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
    os.makedirs(out_dir)
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].cuda()
            targets = sample['transformed_mask'].cuda()
            
            main_out = model(images)
            loss = criterion(main_out, targets)
            running_loss += loss
            
            # 評価
            masks = sample['mask'] # 元サイズのマスク
            mask_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=mask_size, mode='bilinear', align_corners=False).cpu()
            if args.save_mask:
                save_main_out_image(main_out_resized, os.path.join(out_dir, f"{i}.png"))
            
            
            preds = torch.sigmoid(main_out_resized)
            # クラスに不均衡があり、1が少ないので閾値を0.5より低めに設定する
            preds = (preds > args.threshold).float()
            print(torch.sum(preds))
            # accuracy
            correct = torch.sum(preds == masks)
            numel = torch.numel(masks)
            total_correct += correct
            total_numel += numel
            # IoU
            intersection = torch.sum(preds * masks)
            union = torch.sum(preds) + torch.sum(masks) - intersection     
            total_intersection += intersection
            total_union += union
            # Dice
            # intersection = torch.sum(preds * masks)
            joint = torch.sum(preds) + torch.sum(masks)
            # total_intersection += intersection
            total_joint += joint
    
    avg_loss = running_loss / len(dataloader)
    global_accuracy = (total_correct / total_numel).item()
    global_iou = (total_intersection / total_union).item()
    global_dice = ((2. * total_intersection) / total_joint).item()
    
    #print(total_correct, total_numel)
    #print(total_intersection, total_union)
    #print(2. * total_intersection, total_joint)
    
    return avg_loss, global_accuracy, global_iou, global_dice


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
