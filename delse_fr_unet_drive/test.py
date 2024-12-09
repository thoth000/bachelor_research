import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt

from models.fr_unet import FR_UNet as Model
import dataloader.drive_loader as drive
from dataloader.drive_loader import unpad_to_original
from models.lse import *
from models.loss import *

def setup(rank, world_size):
    """DDPの初期設定"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """DDPの終了処理"""
    dist.destroy_process_group()

def check_args(mode='test'):
    parser = argparse.ArgumentParser()
    
    # 必要な引数を設定
    parser.add_argument('--batch', type=int, default=1)  # テストではバッチサイズ1が一般的
    parser.add_argument('--resolution', type=int, default=584)
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--result_name', type=str, default='exp_YYYYMMDD_S')
    parser.add_argument('--save_mask', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='drive', choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])
    parser.add_argument('--transform', type=str, default='standard', choices=['fr_unet', 'standard'])
    parser.add_argument('--pretrained_path', type=str, default='/home/sano/documents/swin_unet_drive/models/swin_tiny_patch4_window7_224.pth')
    
    # モデル固有のパラメータ
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--feature_scale', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fuse', type=bool, default=True)
    parser.add_argument('--out_ave', type=bool, default=True)
    
    parser.add_argument('--dataset_path', type=str, default="/home/sano/dataset/DRIVE")
    parser.add_argument('--dataset_opt', type=str, default="pro")
    
    args = parser.parse_args()
    
    return args


def test_predict(model, dataloader, args, device):
    model.eval()
    out_dir = 'result/predict'
    os.makedirs(out_dir, exist_ok=True)
    # 0ステップ後のレベルセット関数を対象
    total_tp_0 = torch.tensor(0.0, device=device)
    total_tn_0 = torch.tensor(0.0, device=device)
    total_fp_0 = torch.tensor(0.0, device=device)
    total_fn_0 = torch.tensor(0.0, device=device)
    
    # Tステップ後のレベルセット関数を対象
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].to(device)
            
            phi_0, m, vx, vy = model(images)
            phi_T = levelset_evolution(phi_0, vx, vy, m)
            
            masks = sample['mask']
            original_size = masks.shape[2:]
            phi_0_resized = F.interpolate(phi_0, size=original_size, mode='bilinear', align_corners=False).cpu()
            phi_T_resized = F.interpolate(phi_T, size=original_size, mode='bilinear', align_corners=False).cpu()
            
            preds_0 = (phi_0_resized <= args.threshold).float()
            preds_T = (phi_T_resized <= args.threshold).float()
            
            preds_0 = unpad_to_original(preds_0, sample["padding"])
            preds_T = unpad_to_original(preds_T, sample["padding"])
            masks = unpad_to_original(masks, sample["padding"])
            
            # 0ステップ後のレベルセット関数の評価
            total_tp_0 += torch.sum((preds_0 == 1) & (masks == 1)).item()
            total_tn_0 += torch.sum((preds_0 == 0) & (masks == 0)).item()
            total_fp_0 += torch.sum((preds_0 == 1) & (masks == 0)).item()
            total_fn_0 += torch.sum((preds_0 == 0) & (masks == 1)).item()
            
            # Tステップ後のレベルセット関数の評価
            total_tp += torch.sum((preds_T == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds_T == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds_T == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds_T == 0) & (masks == 1)).item()
            
            preds_0 = preds_0.squeeze().numpy()
            preds_0 = (preds_0 * 255).astype(np.uint8)
            preds_T = preds_T.squeeze().numpy()
            preds_T = (preds_T * 255).astype(np.uint8)
            
            if args.rank == 0:
                save_scaler_field(phi_0_resized, os.path.join(out_dir, f'{i+1}_phi_0.png'))
                save_scaler_field(phi_T_resized, os.path.join(out_dir, f'{i+1}_phi_T.png'))
                preds_0_image = Image.fromarray(preds_0)
                preds_0_image.save(os.path.join(out_dir, f'{i+1}_0.png'))
                preds_T_image = Image.fromarray(preds_T)
                preds_T_image.save(os.path.join(out_dir, f'{i+1}_T.png'))
    
    # 結果を集約
    dist.all_reduce(total_tp_0, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn_0, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp_0, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn_0, op=dist.ReduceOp.SUM)
    
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    
    acc_0 = (total_tp_0 + total_tn_0) / (total_tp_0 + total_tn_0 + total_fp_0 + total_fn_0)
    sen_0 = total_tp_0 / (total_tp_0 + total_fn_0) if (total_tp_0 + total_fn_0) > 0 else torch.tensor(0.0, device=device)
    spe_0 = total_tn_0 / (total_tn_0 + total_fp_0) if (total_tn_0 + total_fp_0) > 0 else torch.tensor(0.0, device=device)
    iou_0 = total_tp_0 / (total_tp_0 + total_fp_0 + total_fn_0) if (total_tp_0 + total_fp_0 + total_fn_0) > 0 else torch.tensor(0.0, device=device)
    miou_0 = ((total_tp_0 / (total_tp_0 + total_fp_0 + total_fn_0)) + (total_tn_0 / (total_tn_0 + total_fp_0 + total_fn_0))) / 2
    dice_0 = (2 * total_tp_0) / (2 * total_tp_0 + total_fp_0 + total_fn_0) if (2 * total_tp_0 + total_fp_0 + total_fn_0) > 0 else torch.tensor(0.0, device=device)
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else torch.tensor(0.0, device=device)
    spe = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else torch.tensor(0.0, device=device)
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else torch.tensor(0.0, device=device)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else torch.tensor(0.0, device=device)

    return acc_0.item(), sen_0.item(), spe_0.item(), iou_0.item(), miou_0.item(), dice_0.item(), acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item()

def test(args):
    device = torch.device(f'cuda:{args.rank}')
    
    transform_test = drive.get_transform(args, mode='test')
    testset = drive.DRIVEDataset("test", args.dataset_path, args.dataset_opt, transform = transform_test)
    
    test_sampler = DistributedSampler(testset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    testloader = DataLoader(testset, batch_size=1, sampler=test_sampler, num_workers=args.num_workers)

    model = Model(args).to(device)
    model.load_state_dict(torch.load(args.pretrained_path, map_location=device)['state_dict'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
    
    dist.barrier() # プロセスの同期
    acc_0, sen_0, spe_0, iou_0, miou_0, dice_0, acc, sen, spe, iou, miou, dice = test_predict(model, testloader, args, device)
    
    if args.rank == 0:
        # 0ステップでのレベルセット関数の評価
        print(f'Accuracy_0: {acc_0}')
        print(f'Sensitivity_0: {sen_0}')
        print(f'Specificity_0: {spe_0}')
        print(f'IoU_0: {iou_0}')
        print(f'MIoU_0: {miou_0}')
        print(f'Dice_0: {dice_0}')
        
        # Tステップ後のレベルセット関数の評価
        print(f'Accuracy_T: {acc}')
        print(f'Sensitivity_T: {sen}')
        print(f'Specificity_T: {spe}')
        print(f'IoU_T: {iou}')
        print(f'MIoU_T: {miou}')
        print(f'Dice_T: {dice}')
    
    return acc_0, sen_0, spe_0, iou_0, miou_0, dice_0, acc, sen, spe, iou, miou, dice

def save_scaler_field(output_tensor, filepath, cmap="viridis"):
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


def main():
    args = check_args(mode='test')
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)
    test(args)
    cleanup()

if __name__ == '__main__':
    main()
