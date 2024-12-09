import argparse
import glob
import os
from tqdm import tqdm
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import dataloader.drive_loader as drive

from models.vision_transformer import SwinUnet as Model
from train import train_one_epoch
from evaluate import evaluate
from loss import *

def setup(rank, world_size):
    """DDPの初期設定"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=2))
    torch.cuda.set_device(rank)

def cleanup():
    """DDPの終了処理"""
    dist.destroy_process_group()

def check_args(mode='train'):
    parser = argparse.ArgumentParser()
    
    # 必要な引数を設定
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--exp_dir', type=str, default='exp')
    parser.add_argument('--exp_name', type=str, default='exp_YYYYMMDD_S')
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--save_mask', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='cosine_annealing', choices=['constant', 'cosine_annealing'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--criterion', type=str, default='BCE', choices=['Tversky', 'Focal', 'Dice', 'BCE'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='drive', choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])
    parser.add_argument('--transform', type=str, default='standard', choices=['fr_unet', 'standard'])
    parser.add_argument('--pretrained_path', type=str, default='/home/sano/documents/swin_unet_drive/models/swin_tiny_patch4_window7_224.pth')
    
    args = parser.parse_args()
    
    # dataset
    if args.dataset == 'drive':
        args.num_classes = 1
        # train
        args.image_dir_train = "/home/sano/dataset/drive_rgb/training/images"
        args.mask_dir_train = "/home/sano/dataset/drive_rgb/training/1st_manual"
        # val
        args.image_dir_val = "/home/sano/dataset/drive_rgb/val/images"
        args.mask_dir_val = "/home/sano/dataset/drive_rgb/val/1st_manual"
        # test
        args.image_dir_test = "/home/sano/dataset/drive_rgb/test/images"
        args.mask_dir_test = "/home/sano/dataset/drive_rgb/test/1st_manual"
    
    if mode == 'train':
        # 保存ディレクトリの設定
        args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.exp_dir)
        if not os.path.exists(args.save_dir_root):
            os.makedirs(args.save_dir_root, exist_ok=True)
        args.save_dir = os.path.join(args.save_dir_root, args.exp_name)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        # CSVファイルの設定

        # configの保存
        config_path = os.path.join(args.save_dir, "config.txt")
        save_args_to_file(args, config_path)    

    return args

def save_args_to_file(args, filepath):
    """argsの設定をテキストファイルに保存する関数"""
    with open(filepath, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')    

def train(args):
    setup(args.rank, args.world_size)  # DDP初期化
    
    device = torch.device(f'cuda:{args.rank}')
    
    # transformとデータセットの設定
    transform_train, num_channels = drive.get_transform(args, mode='train')
    transform_val, _ = drive.get_transform(args, mode='val')
    
    trainset = drive.DRIVEDataset(args.image_dir_train, args.mask_dir_train, transform_train)
    valset = drive.DRIVEDataset(args.image_dir_val, args.mask_dir_val, transform_val)

    # DistributedSamplerを使ってデータを分割
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)
    val_sampler = DistributedSampler(valset, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    trainloader = DataLoader(trainset, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=1, sampler=val_sampler, num_workers=0)

    # モデルの準備
    model = Model(args).to(device)
    model.load_from(args)
    model = DDP(model, device_ids=[args.rank])

    # ロス関数、オプティマイザ、スケジューラ設定
    if args.criterion == 'Dice':
        criterion = DiceLoss()
    elif args.criterion == 'Focal':
        criterion = FocalLoss()
    elif args.criterion == 'Tversky':
        criterion = TverskyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=args.eta_min)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    
    # ベスト情報の初期化
    best_info = {
        'epoch': 0,
        'val_loss': 10e10,
        'state_dict': None
    }
    
    # rank=0のみSummaryWriterとtqdmを設定
    writer = None if args.rank != 0 else SummaryWriter(log_dir=args.save_dir)
    progress_bar = tqdm(range(args.max_epoch), ncols=80) if args.rank == 0 else range(args.max_epoch)
    
    for epoch in progress_bar:
        torch.cuda.empty_cache()
        dist.barrier() # プロセスの同期
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device, args.world_size)
        scheduler.step()
        
        if args.rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
        
        if epoch % args.val_interval == args.val_interval - 1:
            val_loss, acc, sen, spe, iou, dice = evaluate(model, valloader, criterion, epoch, args, device)
            if args.rank == 0:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy', acc, epoch)
                writer.add_scalar('Sensitivity', sen, epoch)
                writer.add_scalar('Specificity', spe, epoch)
                writer.add_scalar('IoU', iou, epoch)
                writer.add_scalar('Dice', dice, epoch)
                
                if val_loss < best_info['val_loss']:
                    best_info['epoch'] = epoch
                    best_info['val_loss'] = val_loss
                    best_info['state_dict'] = model.state_dict()

    # モデルの保存 (rank=0のみ)
    if args.rank == 0:
        print('best epoch:', best_info['epoch'])
        torch.save(best_info['state_dict'], os.path.join(args.save_dir, 'final_model.pth'))
        writer.close()

def main():
    args = check_args(mode='train')
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    train(args)
    # mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
    cleanup()  # DDP終了

if __name__ == '__main__':
    main()
