import argparse
import glob
import os
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms


import dataloader.drive_loader as drive

from pspnet import PSPNet
from train import train_one_epoch
from evaluate import evaluate
from loss import *

parser = argparse.ArgumentParser()

# training & testing loop settings
parser.add_argument('--max_epoch', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--batch', type=int, default=16, help='Training batch size')

parser.add_argument('--exp_name', type=str, default='exp')          # folder name of experiment results

parser.add_argument('--val_interval', type=int, default=20, help='Run on val set every args.val_interval epochs')
parser.add_argument('--save_mask', type=bool, default=False, help='if True, save mask when evaluate')
# optimizer settings
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for sigmoid')
parser.add_argument('--criterion', type=str, default='BCE', choices=['Tversky', 'Focal', 'Dice', 'BCE'])
parser.add_argument('--num_workers', type=int, default=4, help='num workers')
# data settings
parser.add_argument('--dataset', type=str, default='drive',
                    choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])

def check_args():
    args = parser.parse_args()
    
    args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.exp_name)
    if not os.path.exists(args.save_dir_root):
        os.makedirs(args.save_dir_root)
    
    runs = sorted(glob.glob(os.path.join(args.save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    args.save_dir = os.path.join(args.save_dir_root, 'run_%03d' % run_id)
    print(args.save_dir)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # save config
    config_path = os.path.join(args.save_dir, "config.txt")
    save_args_to_file(args, config_path)
    
    # num_classes
    if args.dataset == 'drive':
        args.num_classes = 1
    else: # default
        args.num_classes = 1
    
    # dataset
    if args.dataset == 'drive':
        # train
        args.image_dir_train = "/home/sano/dataset/drive/training/images"
        args.mask_dir_train = "/home/sano/dataset/drive/training/1st_manual"
        # val
        args.image_dir_val = "/home/sano/dataset/drive/val/images"
        args.mask_dir_val = "/home/sano/dataset/drive/val/1st_manual"
        # test
        args.image_dir_test = ""
    
    return args    

def save_args_to_file(args, filepath):
    """argsの設定をテキストファイルに保存する関数"""
    with open(filepath, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

def train():
    args = check_args()
    # log setting
    train_log_path = os.path.join(args.save_dir, 'train_log.csv')
    train_header = ['Epoch', 'Loss']
    with open(train_log_path, 'w') as f:
        f.write(','.join(train_header) + '\n')
    
    val_log_path = os.path.join(args.save_dir, 'val_log.csv')
    val_header = ['Epoch', 'Loss', 'Acc', 'IoU', 'Dice']
    with open(val_log_path, 'w') as f:
        f.write(','.join(val_header) + '\n')
    
    # model
    model = PSPNet(args)
    
    # transform
    if args.dataset == 'drive':
        transform_train = transforms.Compose([
            drive.HorizontalFlip(),               # まずはジオメトリ変換を適用
            drive.RandomRotate(),                 # 回転
            drive.RandomShift(),                  # シフト（平行移動）
            drive.RandomScale(),                  # スケール
            # drive.ElasticTransform(),             # 弾性変形は最も大きな変形なので最後に
            drive.HistogramEqualization(),        # 次にヒストグラム平坦化で輝度調整
            drive.RandomBrightnessContrast(),     # 明るさとコントラストのランダム変化
            drive.RandomGamma(),                  # 最後にガンマ変換を適用
            drive.Resize(size=(512, 512)),
            drive.ToTensor()
        ])
        transform_val = transforms.Compose([
            drive.Resize(size=(512, 512)),
            drive.ToTensor()
        ])
    
    # dataset
    if args.dataset == 'drive':
        trainset = drive.DRIVEDataset(args.image_dir_train, args.mask_dir_train, transform_train)
        valset = drive.DRIVEDataset(args.image_dir_val, args.mask_dir_val, transform_val)
    
    # dataloader
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    
    # criterion
    if args.criterion == 'Tversky':
        criterion = TverskyLoss()
    elif args.criterion == 'Focal':
        criterion = FocalLoss()
    elif args.criterion == 'Dice':
        criterion = DiceLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # GPU setting
    model = torch.nn.DataParallel(model).cuda()
    
    # モデル保存用設定
    min_val_loss = 1e+8
    min_val_epoch = 0
    min_val_model = model.state_dict()
    
    for epoch in tqdm(range(args.max_epoch), ncols=80):
        # 1エポック分の学習
        train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
        # logの書き込み
        with open(train_log_path, 'a') as f:
            f.write(f'{epoch}, {train_loss}' + '\n')
        
        if epoch % args.val_interval == args.val_interval - 1:
            # 検証
            val_loss, acc, iou, dice = evaluate(model, valloader, criterion, epoch, args)
            with open(val_log_path, 'a') as f:
                f.write(f'{epoch}, {val_loss}, {acc}, {iou}, {dice}' + '\n')
            # モデル更新
            if val_loss < min_val_loss:
                min_val_epoch = epoch
                min_val_loss = val_loss
                min_val_model = model.state_dict()
    
    # モデル保存
    print(f"best epoch: {min_val_epoch}")
    model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(min_val_model, model_path)


def test():
    pass

if __name__ == '__main__':
    train()