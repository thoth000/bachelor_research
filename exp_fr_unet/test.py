import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from models.fr_unet import FR_UNet as Model
import dataloader.drive_loader as drive
from tqdm import tqdm


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
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--feature_scale', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fuse', type=bool, default=True)
    parser.add_argument('--out_ave', type=bool, default=True)
    
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
    
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = sample['transformed_image'].to(device)
            masks = sample['mask']
            original_size = masks.shape[2:]
            main_out = model(images)
            main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False).cpu()
            preds = (torch.sigmoid(main_out_resized) > args.threshold).float()
            
            soft_skeleton_pred = soft_skeleton(preds)
            soft_skeleton_gt = soft_skeleton(masks)
            
            tp = torch.sum((preds == 1) & (masks == 1)).item()
            tn = torch.sum((preds == 0) & (masks == 0)).item()
            fp = torch.sum((preds == 1) & (masks == 0)).item()
            fn = torch.sum((preds == 0) & (masks == 1)).item()
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            # Dice
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            total_dice += dice
            # cl dice
            tprec = (torch.sum(soft_skeleton_pred * masks) + 1) / (torch.sum(soft_skeleton_pred) + 1)
            tsens = (torch.sum(soft_skeleton_gt * preds) + 1) / (torch.sum(soft_skeleton_gt) + 1)
            cl_dice = (2 * tprec * tsens) / (tprec + tsens)
            total_cl_dice += cl_dice
            
            total_samples += images.size(0)
            
            preds = preds.squeeze().numpy()
            preds = (preds * 255).astype(np.uint8)
            
            pred_image = Image.fromarray(preds)
            pred_image.save(os.path.join(out_dir, f'{i+1}.png'))
    
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_dice, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_cl_dice, op=dist.ReduceOp.SUM)
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else torch.tensor(0.0, device=device)
    spe = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else torch.tensor(0.0, device=device)
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else torch.tensor(0.0, device=device)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = total_dice / total_samples
    cl_dice = total_cl_dice / total_samples

    return acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item(), cl_dice.item()

def test(args):
    device = torch.device(f'cuda:{args.rank}')
    
    transform_test = drive.get_transform(args, mode='test')
    testset = drive.DRIVEDataset("training", args.dataset_path, args.dataset_opt, transform = transform_test)
    
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
