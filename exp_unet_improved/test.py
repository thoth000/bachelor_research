import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from models.unet import UNet as Model
import dataloader.drive_loader as drive
from tqdm import tqdm

import time
from datetime import timedelta

def setup(rank, world_size):
    """DDPの初期設定"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

def cleanup():
    """DDPの終了処理"""
    dist.destroy_process_group()

def check_args(mode='test'):
    parser = argparse.ArgumentParser()
    
    # 必要な引数を設定
    parser.add_argument('--batch', type=int, default=1)  # テストではバッチサイズ1が一般的
    parser.add_argument('--resolution', type=int, default=512)
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
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='elu')
    
    parser.add_argument('--dataset_path', type=str, default="/home/sano/dataset/DRIVE")
    parser.add_argument('--dataset_opt', type=str, default="pro")
    
    args = parser.parse_args()
    
    return args


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


def soft_skeleton(x, thresh_width=50):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    # print("proper soft_skeleton")
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x


def dti(preds, thresh_low=0.3, thresh_high=0.5, max_iter=1000):
    fixed_mask = (preds > thresh_high).float()
    checkable_mask = (preds >= thresh_low).float()
    new_fixed_mask = torch.zeros_like(fixed_mask)
    count = 0
    while not torch.equal(new_fixed_mask, fixed_mask):
        count += 1
        new_fixed_mask = fixed_mask.clone()
        fixed_mask = F.max_pool2d(fixed_mask, 3, 1, 1) * checkable_mask
    
    return fixed_mask


def dti_normal(preds, thresh_low=0.3, thresh_high=0.5, max_iter=1000):
    h, w = preds.shape[2:]
    bin = (preds > thresh_high).float()
    gbin = bin.clone()
    gbin_pre = torch.zeros_like(gbin)
    count = 0
    while not torch.equal(gbin_pre, gbin):
        count += 1
        gbin_pre = gbin.clone()
        for i in range(h):
            for j in range(w):
                if not gbin[0, 0, i, j] and (preds[0, 0, i, j] < thresh_high) and (preds[0, 0, i, j] >= thresh_low):
                    if gbin[0, 0, i-1, j-1] or gbin[0, 0, i-1, j] or gbin[0, 0, i-1, j+1] or gbin[0, 0, i, j-1] or gbin[0, 0, i, j+1] or gbin[0, 0, i+1, j-1] or gbin[0, 0, i+1, j] or gbin[0, 0, i+1, j+1]:
                        gbin[0, 0, i, j] = 1.0
    
    print(count)
    
    return gbin

def test_predict(model, dataloader, args, device):
    model.eval()
    out_dir = 'result/predict'
    os.makedirs(out_dir, exist_ok=True)
    
    total_samples = torch.tensor(0.0, device=device)
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    total_dice = torch.tensor(0.0, device=device)
    total_cl_dice = torch.tensor(0.0, device=device)
    
    total_time = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        tbar = tqdm(enumerate(dataloader), total=len(dataloader)) if args.rank == 0 else enumerate(dataloader)
        
        for i, sample in tbar:
            dist.barrier() # 同期
            images = sample['transformed_image'].to(device)
            masks = sample['mask'].to(device)
            original_size = masks.shape[2:]
            main_out = model(images)
            main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False)
            preds = torch.sigmoid(main_out_resized) # [B, 1, H, W]
            masks_pred = (preds > args.threshold).float()
            
            start = time.time()
            
            masks_dti = dti(preds, thresh_low=0.3, thresh_high=0.5)
            
            end = time.time()
            
            total_time += torch.tensor(end - start, device=device)
            
            masks_eval = masks_pred
            
            soft_skeleton_pred = soft_skeleton(masks_eval)
            soft_skeleton_gt = soft_skeleton(masks)
            
            tp = torch.sum((masks_eval == 1) & (masks == 1)).item()
            tn = torch.sum((masks_eval == 0) & (masks == 0)).item()
            fp = torch.sum((masks_eval == 1) & (masks == 0)).item()
            fn = torch.sum((masks_eval == 0) & (masks == 1)).item()
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            # Dice
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            total_dice += dice
            # cl dice
            tprec = (torch.sum(soft_skeleton_pred * masks) + 1) / (torch.sum(soft_skeleton_pred) + 1)
            tsens = (torch.sum(soft_skeleton_gt * masks_eval) + 1) / (torch.sum(soft_skeleton_gt) + 1)
            cl_dice = (2 * tprec * tsens) / (tprec + tsens)
            total_cl_dice += cl_dice
            
            total_samples += images.size(0)
            
            if True:
                masks = masks.squeeze().cpu().numpy()
                masks = (masks * 255).astype(np.uint8)
                gt_image = Image.fromarray(masks)
                gt_image.save(os.path.join(out_dir, f'gt_{i+1}_{args.rank}.png'))
                
                masks_pred = masks_pred.squeeze().cpu().numpy()
                masks_pred = (masks_pred * 255).astype(np.uint8)
                pred_image = Image.fromarray(masks_pred)
                pred_image.save(os.path.join(out_dir, f'pred_{i+1}_{args.rank}.png'))
                
                masks_dti = masks_dti.squeeze().cpu().numpy()
                masks_dti = (masks_dti * 255).astype(np.uint8)
                dti_image = Image.fromarray(masks_dti)
                dti_image.save(os.path.join(out_dir, f'dti_{i+1}_{args.rank}.png'))

    
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_dice, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_cl_dice, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_time, op=dist.ReduceOp.SUM)
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else torch.tensor(0.0, device=device)
    spe = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else torch.tensor(0.0, device=device)
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else torch.tensor(0.0, device=device)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = total_dice / total_samples
    cl_dice = total_cl_dice / total_samples

    ave_time = total_time / total_samples
    print(ave_time.item())

    return acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item(), cl_dice.item()

def test(args):
    device = torch.device(f'cuda:{args.rank}')
    
    transform_test = drive.get_transform(args, mode='test')
    testset = drive.DRIVEDataset("test", args.dataset_path, args.dataset_opt, transform = transform_test)
    
    test_sampler = DistributedSampler(testset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    testloader = DataLoader(testset, batch_size=1, sampler=test_sampler, num_workers=args.num_workers)

    model = Model(args).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
    
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    acc, sen, spe, iou, miou, dice, cl_dice = test_predict(model, testloader, args, device)
    
    if args.rank == 0:
        print(f'Accuracy: {acc}')
        print(f'Sensitivity: {sen}')
        print(f'Specificity: {spe}')
        print(f'IoU: {iou}')
        print(f'MIoU: {miou}')
        print(f'Dice: {dice}')
        print(f'CL Dice: {cl_dice}')
    
    return acc, sen, spe, iou, miou, dice, cl_dice


def main():
    args = check_args(mode='test')
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)
    test(args)
    cleanup()

if __name__ == '__main__':
    main()
