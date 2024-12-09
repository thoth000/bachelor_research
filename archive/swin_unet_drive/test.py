import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from models.vision_transformer import SwinUnet as Model
import dataloader.drive_loader as drive

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
    
    args = parser.parse_args()
    
    # dataset
    if args.dataset == 'drive':
        args.num_classes = 1
        # test directories
        args.image_dir_test = "/home/sano/dataset/drive/test/images"
        args.mask_dir_test = "/home/sano/dataset/drive/test/1st_manual"
    
    # 保存ディレクトリの設定 (テスト出力用ディレクトリ設定)
    if mode == 'test':
        args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_dir)
        args.save_dir = os.path.join(args.save_dir_root, args.result_name)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
    
    return args


def test_predict(model, dataloader, args):
    model.eval()
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
            masks = sample['mask']
            original_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False).cpu()
            
            preds = (torch.sigmoid(main_out_resized) > args.threshold).float()
            total_tp += torch.sum((preds == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds == 0) & (masks == 1)).item()
            
            preds = preds.squeeze().numpy()
            preds = (preds * 255).astype(np.uint8)
            
            pred_image = Image.fromarray(preds)
            pred_image.save(os.path.join(out_dir, f'{i+1}_refine.png'))
    
    return total_tp, total_tn, total_fp, total_fn

def gather_metrics(total_tp, total_tn, total_fp, total_fn, args):
    tp = torch.tensor(total_tp, device=args.device)
    tn = torch.tensor(total_tn, device=args.device)
    fp = torch.tensor(total_fp, device=args.device)
    fn = torch.tensor(total_fn, device=args.device)
    
    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(fn, op=dist.ReduceOp.SUM)

    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=args.device)
    spe = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0, device=args.device)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else torch.tensor(0.0, device=args.device)
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else torch.tensor(0.0, device=args.device)

    if args.rank == 0:
        print(f'Accuracy: {acc.item()}')
        print(f'Sensitivity: {sen.item()}')
        print(f'Specificity: {spe.item()}')
        print(f'IoU: {iou.item()}')
        print(f'Dice: {dice.item()}')
    
    return acc.item(), sen.item(), spe.item(), iou.item(), dice.item()

def test(args):
    setup(args.rank, args.world_size)
    device = torch.device(f'cuda:{args.rank}')
    args.device = device
    
    transform_test, _ = drive.get_transform(args, mode='test')
    testset = drive.DRIVEDataset(args.image_dir_test, args.mask_dir_test, transform_test)
    
    test_sampler = DistributedSampler(testset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    testloader = DataLoader(testset, batch_size=1, sampler=test_sampler, num_workers=args.num_workers)

    model = Model(args).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    
    total_tp, total_tn, total_fp, total_fn = test_predict(model, testloader, args)
    
    gather_metrics(total_tp, total_tn, total_fp, total_fn, args)
    
    cleanup()


def main():
    args = check_args(mode='test')
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    test(args)

if __name__ == '__main__':
    main()
