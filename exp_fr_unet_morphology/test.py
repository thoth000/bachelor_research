import os
import argparse

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from models.fr_unet import FR_UNet as Model
import dataloader.drive_loader as drive
from tqdm import tqdm
import cv2


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


def connect_broken_vessels(mask, kernel_size=3):
    """
    モルフォロジー処理で切れた血管を接続（ポストプロセス版）。
    
    Parameters:
        mask (torch.Tensor): バイナリマスク（形状: [B, 1, H, W]）。
        kernel_size (int): カーネルのサイズ（奇数）。
        
    Returns:
        torch.Tensor: 接続処理後のバイナリマスク（形状: [B, 1, H, W]）。
    """
    # カーネル作成（楕円形）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 結果を格納するリスト
    connected_masks = []
    
    # バッチ内の各サンプルについて処理
    for sample in mask:
        sample_np = sample.squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)
        # モルフォロジー処理
        connected_sample = cv2.morphologyEx(sample_np, cv2.MORPH_CLOSE, kernel)
        connected_masks.append(torch.tensor(connected_sample, dtype=torch.float32))

    # 結果を結合して戻す
    return torch.stack(connected_masks).unsqueeze(1).to(mask.device)


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
            preds = torch.sigmoid(main_out_resized) # [B, 1, H, W]
            masks_pred = (preds > args.threshold).float()
            masks_connected = connect_broken_vessels(masks_pred)
            
            masks_eval = masks_connected
            
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
            
            if args.rank == 0:
                masks_pred = masks_pred.squeeze().numpy()
                masks_pred = (masks_pred * 255).astype(np.uint8)
            
                pred_image = Image.fromarray(masks_pred)
                pred_image.save(os.path.join(out_dir, f'{i+1}.png'))
                
                masks_connected = masks_connected.squeeze().numpy()
                masks_connected = (masks_connected * 255).astype(np.uint8)
                
                connected_image = Image.fromarray(masks_connected)
                connected_image.save(os.path.join(out_dir, f'connected_{i+1}.png'))
            
                
    
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

import matplotlib.pyplot as plt
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


def main():
    args = check_args(mode='test')
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)
    test(args)
    cleanup()

if __name__ == '__main__':
    main()
