import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from models.fr_unet import FR_UNet as Model
from models.lse import select_pde
import dataloader.drive_loader as drive

def setup(rank, world_size):
    """DDPの初期設定"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """DDPの終了処理"""
    dist.destroy_process_group()

def check_args():
    parser = argparse.ArgumentParser()

    # 必要な引数を設定
    parser.add_argument('--model_name', type=str, default='FR_UNet')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--save_dir_root', type=str, default='result')
    parser.add_argument('--save_name', type=str, default='exp_YYYYMMDD_S')
    parser.add_argument('--save_mask', action='store_true', help='If specified, save predicted mask when evaluate')  # 指定すればTrue
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained_path', type=str, default='pretrained.pth')

    # モデル固有のパラメータ
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--feature_scale', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fuse', type=bool, default=True)
    parser.add_argument('--out_ave', type=bool, default=True)
    
    # chのパラメータ
    parser.add_argument('--pde', type=int, default=0)
    parser.add_argument('--D', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=100)
    
    
    # datasetのパス
    parser.dataset_path = parser.add_argument('--dataset_path', type=str, default="/home/sano/dataset/DRIVE")
    parser.dataset_opt = parser.add_argument('--dataset_opt', type=str, default="pro")
    
    args = parser.parse_args()

    args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_dir_root)
    if not os.path.exists(args.save_dir_root):
        os.makedirs(args.save_dir_root, exist_ok=True)
    args.save_dir = os.path.join(args.save_dir_root, args.save_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    return args


def test_predict(model, dataloader, args, device):
    model.eval()
    
    origin_tp = torch.tensor(0.0, device=device)
    origin_tn = torch.tensor(0.0, device=device)
    origin_fp = torch.tensor(0.0, device=device)
    origin_fn = torch.tensor(0.0, device=device)
    
    ch_tp = torch.tensor(0.0, device=device)
    ch_tn = torch.tensor(0.0, device=device)
    ch_fp = torch.tensor(0.0, device=device)
    ch_fn = torch.tensor(0.0, device=device)
    
    tbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=80) if args.rank == 0 else enumerate(dataloader)
    
    with torch.no_grad():
        for i, sample in tbar:
            images = sample['transformed_image'].to(device)
            
            main_out = model(images)
            masks = sample['mask']
            original_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False).cpu()
            
            sigmoid_out = torch.sigmoid(main_out_resized) * 2 - 1
            
            # sigmoid_outに対してKahn-Hilliardを適用
            sigmoid_out_np = sigmoid_out.numpy()
            
            # select pde
            pde = select_pde(args.pde)
            if args.rank == 0 and i < 2:
                ch_out_np = pde(sigmoid_out_np, D=args.D, gamma=args.gamma, dt=args.dt, steps=args.steps, tqdm_log=True, make_gif=True, gif_path=os.path.join(args.save_dir, f'{i+1}.gif'))
            else:
                ch_out_np = pde(sigmoid_out_np, D=args.D, gamma=args.gamma, dt=args.dt, steps=args.steps)
            ch_out = torch.tensor(ch_out_np)
            
            # origin evaluation
            preds = (sigmoid_out > args.threshold).float()
            origin_tp += torch.sum((preds == 1) & (masks == 1)).item()
            origin_tn += torch.sum((preds == 0) & (masks == 0)).item()
            origin_fp += torch.sum((preds == 1) & (masks == 0)).item()
            origin_fn += torch.sum((preds == 0) & (masks == 1)).item()

            # ch evaluation
            ch_preds = (ch_out > args.threshold).float()
            ch_tp += torch.sum((ch_preds == 1) & (masks == 1)).item()
            ch_tn += torch.sum((ch_preds == 0) & (masks == 0)).item()
            ch_fp += torch.sum((ch_preds == 1) & (masks == 0)).item()
            ch_fn += torch.sum((ch_preds == 0) & (masks == 1)).item()
            
            if args.rank == 0 and args.save_mask and i < 2:
                visualize_and_save_scalar_field(sigmoid_out_np[0, 0], os.path.join(args.save_dir, f'{i+1}_origin.png'))
                visualize_and_save_scalar_field(ch_out[0, 0], os.path.join(args.save_dir, f'{i+1}_ch.png'))
    
    dist.all_reduce(origin_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(origin_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(origin_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(origin_fn, op=dist.ReduceOp.SUM)
    
    dist.all_reduce(ch_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(ch_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(ch_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(ch_fn, op=dist.ReduceOp.SUM)
    
    acc = (origin_tp + origin_tn) / (origin_tp + origin_tn + origin_fp + origin_fn)
    sen = origin_tp / (origin_tp + origin_fn) if (origin_tp + origin_fn) > 0 else torch.tensor(0.0, device=device)
    spe = origin_tn / (origin_tn + origin_fp) if (origin_tn + origin_fp) > 0 else torch.tensor(0.0, device=device)
    iou = origin_tp / (origin_tp + origin_fp + origin_fn) if (origin_tp + origin_fp + origin_fn) > 0 else torch.tensor(0.0, device=device)
    miou = ((origin_tp / (origin_tp + origin_fp + origin_fn)) + (origin_tn / (origin_tn + origin_fp + origin_fn))) / 2
    dice = (2 * origin_tp) / (2 * origin_tp + origin_fp + origin_fn) if (2 * origin_tp + origin_fp + origin_fn) > 0 else torch.tensor(0.0, device=device)
    
    acc_ch = (ch_tp + ch_tn) / (ch_tp + ch_tn + ch_fp + ch_fn)
    sen_ch = ch_tp / (ch_tp + ch_fn) if (ch_tp + ch_fn) > 0 else torch.tensor(0.0, device=device)
    spe_ch = ch_tn / (ch_tn + ch_fp) if (ch_tn + ch_fp) > 0 else torch.tensor(0.0, device=device)
    iou_ch = ch_tp / (ch_tp + ch_fp + ch_fn) if (ch_tp + ch_fp + ch_fn) > 0 else torch.tensor(0.0, device=device)
    miou_ch = ((ch_tp / (ch_tp + ch_fp + ch_fn)) + (ch_tn / (ch_tn + ch_fp + ch_fn))) / 2
    dice_ch = (2 * ch_tp) / (2 * ch_tp + ch_fp + ch_fn) if (2 * ch_tp + ch_fp + ch_fn) > 0 else torch.tensor(0.0, device=device)
    
    return acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item(), acc_ch.item(), sen_ch.item(), spe_ch.item(), iou_ch.item(), miou_ch.item(), dice_ch.item()

def test(args):
    device = torch.device(f'cuda:{args.rank}')
    
    transform_test = drive.get_transform(args, mode='test')
    testset = drive.DRIVEDataset("test", args.dataset_path, args.dataset_opt, transform = transform_test)
    
    test_sampler = DistributedSampler(testset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    testloader = DataLoader(testset, batch_size=1, sampler=test_sampler, num_workers=args.num_workers)

    model = Model(args).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
    
    checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    acc, sen, spe, iou, miou, dice, acc_ch, sen_ch, spe_ch, iou_ch, miou_ch, dice_ch = test_predict(model, testloader, args, device)
    
    return acc, sen, spe, iou, miou, dice, acc_ch, sen_ch, spe_ch, iou_ch, miou_ch, dice_ch


def visualize_and_save_scalar_field(scalar_field, output_path, cmap="RdBu"):
    """
    Visualize and save a scalar field in the range [-1, 1] using matplotlib.

    Parameters:
        scalar_field (numpy.ndarray): Input scalar field with values in the range [-1, 1].
        output_path (str): Path to save the output image.
        cmap (str): Colormap to use for visualization (default: "RdBu").
    """
    # assert np.all(scalar_field >= -1) and np.all(scalar_field <= 1), "Scalar field must be in the range [-1, 1]"

    # Create the figure and axis
    plt.figure(figsize=(6, 6))
    plt.imshow(scalar_field, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label="Scalar value [-1, 1]")

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def main():
    args = check_args()
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)
    acc, sen, spe, iou, miou, dice, acc_ch, sen_ch, spe_ch, iou_ch, miou_ch, dice_ch = test(args)
    # args.save_dirにmetricsを保存
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Accuracy: {acc_ch}\n')
        f.write(f'Sensitivity: {sen_ch}\n')
        f.write(f'Specificity: {spe_ch}\n')
        f.write(f'IoU: {iou_ch}\n')
        f.write(f'MIoU: {miou_ch}\n')
        f.write(f'Dice: {dice_ch}\n')
    
    cleanup()

if __name__ == '__main__':
    main()
