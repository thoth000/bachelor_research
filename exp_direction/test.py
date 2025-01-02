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
from models.fr_unet import Anisotropic_Diffusion_OnlyDiv as Model
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
    
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    tbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=80) if args.rank == 0 else enumerate(dataloader)
    
    with torch.no_grad():
        for i, sample in tbar:
            images = sample['transformed_image'].to(device)
            
            main_out, s_long, s_short, v_long = model(images)
            masks = sample['mask']
            original_size = masks.shape[2:]
            main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False).cpu()
            
            # origin evaluation
            preds = (main_out_resized > args.threshold).float()
            total_tp += torch.sum((preds == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds == 0) & (masks == 1)).item()
            
            if args.rank == 0 and args.save_mask and i < 5:
                save_scalar_field_as_image(main_out_resized[0, 0], os.path.join(args.save_dir, f'{i+1}_origin.png'))
                save_scalar_field_as_image(preds[0, 0], os.path.join(args.save_dir, f'{i+1}_binary.png'))
    
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else torch.tensor(0.0, device=device)
    spe = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else torch.tensor(0.0, device=device)
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else torch.tensor(0.0, device=device)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else torch.tensor(0.0, device=device)

    return acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item()

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
    
    acc, sen, spe, iou, miou, dice= test_predict(model, testloader, args, device)
    
    return acc, sen, spe, iou, miou, dice


def visualize_and_save_scalar_field(scalar_field, output_path, cmap="gray"):
    """
    Visualize and save a scalar field in the range [-1, 1] using matplotlib.

    Parameters:
        scalar_field (numpy.ndarray): Input scalar field with values in the range [-1, 1].
        output_path (str): Path to save the output image.
        cmap (str): Colormap to use for visualization.
    """

    # Create the figure and axis
    plt.figure(figsize=(6, 6))
    plt.imshow(scalar_field, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label="Scalar value [0, 1]")

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def save_scalar_field_as_image(scalar_field, output_path):
    scalar_field = scalar_field.detach().cpu().numpy()
    # スカラー場が [0, 1] の範囲内にあることを確認
    if scalar_field.min() < 0 or scalar_field.max() > 1:
        raise ValueError("The scalar field values must be in the range [0, 1].")

    # 0-1の値を0-255のグレースケールにスケーリング
    scalar_field_scaled = (scalar_field * 255).astype(np.uint8)

    # Imageに変換して保存
    image = Image.fromarray(scalar_field_scaled, mode="L")  # "L"は8-bit grayscale
    image.save(output_path)


def main():
    args = check_args()
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)
    acc, sen, spe, iou, miou, dice = test(args)
    # args.save_dirにmetricsを保存
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Accuracy: {acc}\n')
        f.write(f'Sensitivity: {sen}\n')
        f.write(f'Specificity: {spe}\n')
        f.write(f'IoU: {iou}\n')
        f.write(f'MIoU: {miou}\n')
        f.write(f'Dice: {dice}\n')
    
    cleanup()

if __name__ == '__main__':
    main()
