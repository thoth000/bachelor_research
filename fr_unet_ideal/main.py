import argparse
import os
import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
import numpy as np

import dataloader.drive_loader as drive
from models.fr_unet import FR_UNet as Model
from evaluate import evaluate
from loss import *

from train import train
from test import test


def setup(rank, world_size):
    """DDPの初期設定"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=2))
    torch.cuda.set_device(rank)


def cleanup():
    """DDPの終了処理"""
    dist.destroy_process_group()


def check_args():
    parser = argparse.ArgumentParser()

    # 必要な引数を設定
    parser.add_argument('--model_name', type=str, default='FR_UNet')
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--exp_dir', type=str, default='exp')
    parser.add_argument('--exp_name', type=str, default='exp_YYYYMMDD_S')
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--save_mask', action='store_true', help='If specified, save predicted mask when evaluate')  # 指定すればTrue
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='cosine_annealing', choices=['constant', 'cosine_annealing'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--criterion', type=str, default='BCE', choices=['Tversky', 'Focal', 'Dice', 'BCE', 'BalancedBCE'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='drive',
                        choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])
    parser.add_argument('--transform', type=str, default='fr_unet', choices=['fr_unet', 'standard'])

    # モデル固有のパラメータ
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--feature_scale', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fuse', type=bool, default=True)
    parser.add_argument('--out_ave', type=bool, default=True)
    
    # datasetのパス
    parser.dataset_path = parser.add_argument('--dataset_path', type=str, default="/home/sano/dataset/DRIVE")
    parser.dataset_opt = parser.add_argument('--dataset_opt', type=str, default="pro")
    
    args = parser.parse_args()

    args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.exp_dir)
    if not os.path.exists(args.save_dir_root):
        os.makedirs(args.save_dir_root, exist_ok=True)
    args.save_dir = os.path.join(args.save_dir_root, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # configの保存
    config_path = os.path.join(args.save_dir, "config.txt")
    save_args_to_file(args, config_path)

    # dataset
    if args.dataset == 'drive':
        args.num_classes = 1
        # train
        args.image_dir_train = "/home/sano/dataset/drive_pro/training/images"
        args.mask_dir_train = "/home/sano/dataset/drive_pro/training/1st_manual"
        # val
        args.image_dir_val = "/home/sano/dataset/drive_pro/val/images"
        args.mask_dir_val = "/home/sano/dataset/drive_pro/val/1st_manual"
        # test
        args.image_dir_test = "/home/sano/dataset/drive_pro/test/images"
        args.mask_dir_test = "/home/sano/dataset/drive_pro/test/1st_manual"

    return args


def save_args_to_file(args, filepath):
    """argsの設定をテキストファイルに保存する関数"""
    with open(filepath, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

def main():
    args = check_args()
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)

    if args.rank == 0: # rank0のみtensorboardを使う
        writer = SummaryWriter(log_dir=args.save_dir)
        train(args, writer)
        args.pretrained_path = os.path.join(args.save_dir, "final_model.pth")
        acc, sen, spe, iou, miou, dice = test(args)
        metrics = {'acc': acc, 'sen': sen, 'spe': spe, 'iou': iou, 'miou': miou, 'dice': dice}
        with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
            for k, v in metrics.items():
                f.write(f'{k}: {v})\n')
        # writer.add_hparams(vars(args), metrics)
        writer.close()
    else: # rank0以外はtrainとtestを実行
        train(args)
        args.pretrained_path = os.path.join(args.save_dir, "final_model.pth")
        acc, sen, spe, iou, miou, dice = test(args)
    

if __name__ == '__main__':
    main()
