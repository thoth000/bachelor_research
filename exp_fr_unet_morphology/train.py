import argparse
import os
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import dataloader.drive_loader as drive

from models.fr_unet import FR_UNet as Model
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
    parser.add_argument('--save_mask', action='store_true', help='If specified, save predicted mask when evaluate') # 指定すればTrue
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='cosine_annealing', choices=['constant', 'cosine_annealing'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--criterion', type=str, default='BCE', choices=['Tversky', 'Focal', 'Dice', 'BCE'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='drive', choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])
    parser.add_argument('--transform', type=str, default='standard', choices=['fr_unet', 'standard'])
    
    # モデル固有のパラメータ
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--feature_scale', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fuse', type=bool, default=True)
    parser.add_argument('--out_ave', type=bool, default=True)
    
    
    args = parser.parse_args()
    
    # dataset
    if args.dataset == 'drive':
        args.num_classes = 1
        # train
        args.image_dir_train = "/home/sano/dataset/drive/training/images"
        args.mask_dir_train = "/home/sano/dataset/drive/training/1st_manual"
        # val
        args.image_dir_val = "/home/sano/dataset/drive/val/images"
        args.mask_dir_val = "/home/sano/dataset/drive/val/1st_manual"
        # test
        args.image_dir_test = "/home/sano/dataset/drive/test/images"
        args.mask_dir_test = "/home/sano/dataset/drive/test/1st_manual"
    
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
    return F.max_pool2d(input*-1, kernel_size, stride, padding)*-1 # 最大プーリングを適用して再度反転


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


def train_one_epoch(args, model, dataloader, criterion, optimizer, device, world_size):
    model.train()
    train_loss = torch.tensor(0.0, device=device)  # train_lossをテンソルで初期化

    num_sample = torch.tensor(0, device=device)  # サンプル数をテンソルで初期化

    optimizer.zero_grad()
    
    tbar = tqdm(enumerate(dataloader), ncols=80, total=len(dataloader)) if args.rank == 0 else enumerate(dataloader)
    
    for i, sample in tbar:
        images = sample['transformed_image'].to(device)
        masks = sample['transformed_mask'].to(device)
        
        num_sample += images.size(0)  # サンプル数をカウント
        
        preds = model(images)
        preds = torch.sigmoid(preds)
        soft_skeleton_pred = soft_skeleton(preds)
        soft_skeleton_gt = soft_skeleton(masks)
        
        # loss
        loss = Loss()(preds, masks, soft_skeleton_pred, soft_skeleton_gt, alpha=args.alpha)
        loss.backward()
        
        if num_sample.item() >= 64 / 4:
            optimizer.step()
            optimizer.zero_grad()
            num_sample.zero_()
        
        train_loss += loss  # 累積

    # DDP: 全プロセスで損失を集約
    if dist.is_initialized():
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss /= world_size  # プロセス数で割って平均化

    # optimizer.step()  # 勾配を適用

    avg_loss = train_loss.item() / len(dataloader)  # 最終的にスカラー化してからバッチ数で平均
    return avg_loss

def train(args, writer=None):
    device = torch.device(f'cuda:{args.rank}')
    
    # transformとデータセットの設定
    transform_train = drive.get_transform(args, mode='training')
    transform_val = drive.get_transform(args, mode='val')
    
    trainset = drive.DRIVEDataset("training", args.dataset_path, args.dataset_opt ,is_val = False, transform = transform_train)
    valset = drive.DRIVEDataset("test", args.dataset_path, args.dataset_opt, is_val = True, transform = transform_val)

    # DistributedSamplerを使ってデータを分割
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)
    val_sampler = DistributedSampler(valset, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    trainloader = DataLoader(trainset, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=1, sampler=val_sampler, num_workers=0)

    # モデルの準備
    model = Model(args).to(device)
    model = DDP(model, device_ids=[args.rank], find_unused_parameters=True)

    # ロス関数、オプティマイザ、スケジューラ設定
    if args.criterion == 'Dice':
        criterion = DiceLoss()
    elif args.criterion == 'Focal':
        criterion = FocalLoss()
    elif args.criterion == 'Tversky':
        criterion = TverskyLoss()
    elif args.criterion == 'BalancedBCE':
        criterion = BalancedBinaryCrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=args.eta_min)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    
    # ベスト情報の初期化
    best_info = {
        'epoch': 0,
        'Dice': 0,
        'state_dict': None
    }
    
    # rank=0のみtqdmを設定
    progress_bar = tqdm(range(args.max_epoch), ncols=80) if args.rank == 0 else range(args.max_epoch)
    
    for epoch in progress_bar:
        torch.cuda.empty_cache()
        dist.barrier() # プロセスの同期
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(args, model, trainloader, criterion, optimizer, device, args.world_size)
        scheduler.step()
        
        if args.rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
        
        if epoch % args.val_interval == args.val_interval - 1:
            val_loss, acc, sen, spe, iou, miou, dice, cl_dice = evaluate(model, valloader, criterion, epoch, args, device)
            if args.rank == 0:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy', acc, epoch)
                writer.add_scalar('Sensitivity', sen, epoch)
                writer.add_scalar('Specificity', spe, epoch)
                writer.add_scalar('IoU', iou, epoch)
                writer.add_scalar('MIoU', miou, epoch)
                writer.add_scalar('Dice', dice, epoch)
                writer.add_scalar('CL Dice', cl_dice, epoch)
                
                if dice > best_info['Dice']:
                    print(f"\nBest model found at epoch {epoch + 1}\n")
                    best_info['epoch'] = epoch
                    best_info['Dice'] = dice
                    best_info['state_dict'] = model.state_dict()

    # モデルの保存 (rank=0のみ)
    if args.rank == 0:
        print('best epoch:', best_info['epoch'])
        best_info['args'] = vars(args)
        torch.save(best_info, os.path.join(args.save_dir, 'final_model.pth'))


def main():
    args = check_args(mode='train')
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    setup(args.rank, args.world_size)  # DDP初期化
    writer = None if args.rank != 0 else SummaryWriter(log_dir=args.save_dir)
    train(args, writer)
    writer.close() if args.rank == 0 else None
    cleanup()  # DDP終了

if __name__ == '__main__':
    main()
