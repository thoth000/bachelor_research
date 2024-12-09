import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def test_predict(model, dataloader, args):
    model.eval()
    # ディレクトリ生成
    out_dir = 'result/predict'
    os.makedirs(out_dir, exist_ok=True)
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].cuda()
            
            main_out = model(images)
            # 評価
            masks = sample['mask'] # 元サイズのマスク
            original_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False).cpu()
            
            preds = torch.sigmoid(main_out_resized)
            preds = (preds > args.threshold).float()
            # tp, tn, fp, fn
            total_tp += torch.sum((preds == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds == 0) & (masks == 1)).item()
            
            preds = preds.squeeze().numpy()  # 予測マスクをNumPy配列に変換
            
            # predsを0-255の範囲に変換
            preds = (preds * 255).astype(np.uint8)
            
            # 画像を保存
            pred_image = Image.fromarray(preds)
            pred_image.save(os.path.join(out_dir, f'{i+1}.png'))
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn)
    spe = total_tn / (total_tn + total_fp)
    iou = total_tp / (total_tp + total_fp + total_fn)
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) # F1
    
    return acc, sen, spe, iou, dice

