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
            
            # if args.double_thresh:
            #    preds = double_threshold_iteration(main_out_resized, h_thresh=args.threshold, l_thresh=args.threshold_low)      
            # else:
            preds = (torch.sigmoid(main_out_resized) > args.threshold).float()
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
            pred_image.save(os.path.join(out_dir, f'{i+1}_refine.png'))
            
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn)
    spe = total_tn / (total_tn + total_fp)
    iou = total_tp / (total_tp + total_fp + total_fn)
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) # F1
    
    return acc, sen, spe, iou, dice

# 二重閾値
def double_threshold_iteration(logits, h_thresh, l_thresh):
    assert logits.shape[0] == 1 and logits.shape[1] == 1, "入力は[1, 1, H, W]の形式である必要があります"
    logits = logits.squeeze(0).squeeze(0)  # [H, W]に変換
    
    h, w = logits.shape
    probs = np.array(torch.sigmoid(logits).cpu().detach(), dtype=np.float32)  # 0〜1の範囲の値のまま使用
    bin = np.where(probs >= h_thresh, 1, 0).astype(np.uint8)  # 二値化
    gbin = bin.copy()
    gbin_pre = gbin - 1

    # 反復処理
    while (gbin_pre != gbin).any():
        gbin_pre = gbin.copy()
        for i in range(1, h-1):  # 境界チェック
            for j in range(1, w-1):
                if gbin[i][j] == 0 and h_thresh > probs[i][j] >= l_thresh:
                    if (gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or 
                        gbin[i][j-1] or gbin[i][j+1] or 
                        gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]):
                        gbin[i][j] = 1

    # Tensorに変換して返す
    return torch.tensor(gbin[np.newaxis, np.newaxis, :, :], dtype=torch.float32)


