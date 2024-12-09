import argparse
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

from models.fr_unet import FR_UNet as Model
from models.lse import *
from models.loss import *
from evaluate import evaluate

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

def add_module_prefix(state_dict):
    return {f"module.{k}": v for k, v in state_dict.items()}

def train_one_epoch(epoch, model, dataloader, optimizer, args, device):
    model.train()
    train_loss = torch.tensor(0.0, device=device)  # train_lossをテンソルで初期化

    total_samples = torch.tensor(0.0, device=device)

    for i, sample in enumerate(dataloader):
        images = sample['transformed_image'].to(device)

        optimizer.zero_grad()
        phi_0, m, vx, vy = model(images)
        
        # 初期レベルセットのロス計算
        if epoch < args.pre_epoch or args.pre_pass:
            loss_0 = level_map_loss(phi_0, sample['sdt'].to(device), alpha=1)
            phi_T = levelset_evolution(phi_0 + (10 * np.random.rand() - 5), vx, vy, m)
            vy_gt, vx_gt = gradient_sobel(sample['dt'].to(device), split=True)
            loss_direct = vector_field_loss(vx, vy, vx_gt, vy_gt)
            loss_T = LSE_output_loss(phi_T, sample['transformed_mask'].to(device))
            # print('phi_0:', loss_0.item(), 'phi_T:', loss_T.item(), 'direct:', loss_direct.item())
            loss = loss_0 + loss_direct + loss_T 
        else:
            phi_T = levelset_evolution(phi_0 + (10 * np.random.rand() - 5), vx, vy, m)
            loss_T = LSE_output_loss(phi_T, sample['transformed_mask'].to(device))
            loss = loss_T  
        
        loss.backward()
        optimizer.step()

        train_loss += loss  # 累積
        total_samples += images.size(0)  # バッチ内のサンプル数を加算

    # DDP: 全プロセスで損失を集約
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    train_loss /= total_samples  # バッチ数で平均

    return train_loss.item()

def train(args, writer=None):
    device = torch.device(f'cuda:{args.rank}')
    
    # transformとデータセットの設定
    transform_train = drive.get_transform(args, mode='training')
    transform_val = drive.get_transform(args, mode='val')
    
    trainset = drive.DRIVEDataset("training", args.dataset_path, args.dataset_opt ,is_val = False, split=0.9, transform = transform_train)
    valset = drive.DRIVEDataset("training", args.dataset_path, args.dataset_opt, is_val = True, split=0.9, transform = transform_val)

    # DistributedSamplerを使ってデータを分割
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)
    val_sampler = DistributedSampler(valset, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    trainloader = DataLoader(trainset, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=1, sampler=val_sampler, num_workers=0)

    # モデルの準備
    model = Model(args).to(device)
    model = DDP(model, device_ids=[args.rank], find_unused_parameters=True)
    
    if args.pretrained_path is not None: # 事前学習済みモデルがある場合、optimizerで最終層のみを更新
        # モデル状態を取得
        current_state_dict = model.state_dict()
        
        pretrained_state_dict = torch.load(args.pretrained_path, weights_only=False)['state_dict']
        pretrained_state_dict = add_module_prefix(pretrained_state_dict)
        
        # フィルタリングして、一致する層だけを更新
        # 一致する層だけを更新する辞書
        filtered_state_dict = {
            k: v for k, v in pretrained_state_dict.items()
            if k in current_state_dict and v.size() == current_state_dict[k].size()
        }

        # 一致しない層を格納するリスト
        delse_layers = [k for k in current_state_dict if k not in filtered_state_dict]
        
        # 事前学習済みモデルを用いて一致層を更新
        current_state_dict.update(filtered_state_dict)
        model.load_state_dict(current_state_dict)
        
        if args.fix_pretrained_params: # 訓練済みモデルのパラメータを固定
            # 全てのパラメータを固定 (requires_grad=False)
            for param in model.parameters():
                param.requires_grad = False
        
            # DELSEのみのパラメータを学習可能に (requires_grad=True)
            for name, param in model.named_parameters():
                if name in delse_layers:
                    param.requires_grad = True
            delse_params = filter(lambda p: p.requires_grad, model.parameters())
        
            # optimizerの設定
            optimizer = optim.Adam(delse_params, lr=args.lr, weight_decay=args.weight_decay)
        else: # 訓練済みモデルのパラメータを固定しない
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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
    
    # rank=0のみtqdmを設定
    progress_bar = tqdm(range(args.max_epoch), ncols=80) if args.rank == 0 else range(args.max_epoch)
    
    for epoch in progress_bar:
        torch.cuda.empty_cache()
        dist.barrier() # プロセスの同期
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(epoch, model, trainloader, optimizer, args, device)
        scheduler.step()
        
        if args.rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
        
        if epoch % args.val_interval == args.val_interval - 1:
            val_loss, acc_0, sen_0, spe_0, iou_0, miou_0, dice_0, acc, sen, spe, iou, miou, dice = evaluate(model, valloader, epoch, args, device)
            if args.rank == 0:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/0', acc_0, epoch)
                writer.add_scalar('Sensitivity/0', sen_0, epoch)
                writer.add_scalar('Specificity/0', spe_0, epoch)
                writer.add_scalar('IoU/0', iou_0, epoch)
                writer.add_scalar('mIoU/0', miou_0, epoch)
                writer.add_scalar('Dice/0', dice_0, epoch)
                
                writer.add_scalar('Accuracy/T', acc, epoch)
                writer.add_scalar('Sensitivity/T', sen, epoch)
                writer.add_scalar('Specificity/T', spe, epoch)
                writer.add_scalar('IoU/T', iou, epoch)
                writer.add_scalar('mIoU/T', miou, epoch)
                writer.add_scalar('Dice/T', dice, epoch)
                
                if val_loss < best_info['val_loss'] and (epoch > args.pre_epoch or args.pre_pass):
                    print(f'Best model updated! (epoch: {epoch})')
                    best_info['epoch'] = epoch
                    best_info['val_loss'] = val_loss
                    best_info['state_dict'] = model.module.state_dict()

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
