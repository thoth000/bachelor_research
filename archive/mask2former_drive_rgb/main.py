import argparse
import glob
import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader


import dataloader.drive_loader as drive

from models.mask2former import Mask2FormerSegmentation
from train import train_one_epoch
from evaluate import evaluate
from test_predict import test_predict
from loss import *

parser = argparse.ArgumentParser()

# training & testing loop settings
parser.add_argument('--max_epoch', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--batch', type=int, default=3, help='Training batch size')
parser.add_argument('--resolution', type=int, default=3*256, help='image size after transform')

parser.add_argument('--exp_name', type=str, default='exp')          # folder name of experiment results

parser.add_argument('--val_interval', type=int, default=10, help='Run on val set every args.val_interval epochs')
parser.add_argument('--save_mask', type=bool, default=False, help='If it is True, save predicted mask when evaluate')
# adam(optimizer) settings
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Adam\'s weight decay')
# scheduler settings
parser.add_argument('--scheduler', type=str, default='cosine_annealing', choices=['constant', 'cosine_annealing'])

parser.add_argument('--threshold', type=float, default=0.5, help='threshold for sigmoid')
parser.add_argument('--threshold_low', type=float, default=0.3, help='low threshold for double threshold iteration')
parser.add_argument('--criterion', type=str, default='BCE', choices=['Tversky', 'Focal', 'Dice', 'BCE'])
parser.add_argument('--num_workers', type=int, default=4, help='num workers')
# data settings
parser.add_argument('--dataset', type=str, default='drive',
                    choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])
# transform settings
parser.add_argument('--transform', type=str, default='standard', choices=['fr_unet', 'standard'])

# model settings
parser.add_argument('--token_path', type=str, default='/home/sano/documents/mask2former_drive/models/mask2former_token', help='file path: Huggingface token is only written in the file')
parser.add_argument('--pretrained_path', type=str, default='facebook/mask2former-swin-base-ade-semantic', help='pretrained model or image processer path for Mask2Former')

def check_args(mode='train'):
    args = parser.parse_args()
    
    if mode == 'train':
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
        args.image_dir_test = "/home/sano/dataset/drive/test/images"
        args.mask_dir_test = "/home/sano/dataset/drive/test/1st_manual"
    
    return args    

def save_args_to_file(args, filepath):
    """argsの設定をテキストファイルに保存する関数"""
    with open(filepath, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')    

def train():
    args = check_args()
    # hparams
    hparams = {
        'max_epoch': args.max_epoch,
        'batch': args.batch,
        'learning_rate' : args.lr,
        'criterion': args.criterion,
        'scheduler': args.scheduler,
        'transform': args.transform
    }
    
    
    print("train settings")
    print("epoch:", args.max_epoch)
    print("lr:", args.lr)
    print("criterion:", args.criterion)
    print("transform:", args.transform)
    print("scheduler:", args.scheduler)
    
    # log setting
    train_log_path = os.path.join(args.save_dir, 'train_log.csv')
    train_header = ['Epoch', 'Loss']
    with open(train_log_path, 'w') as f:
        f.write(','.join(train_header) + '\n')
    
    val_log_path = os.path.join(args.save_dir, 'val_log.csv')
    val_header = ['Epoch', 'Loss', 'Acc', 'Sen', 'Spe', 'IoU', 'Dice']
    with open(val_log_path, 'w') as f:
        f.write(','.join(val_header) + '\n')
    
    # transform
    if args.dataset == 'drive':
        transform_train, num_channels = drive.get_transform(args, mode='train')
        transform_val, _ = drive.get_transform(args, mode='val')
    args.num_channels = num_channels # argsにも追加
    
    # dataset
    if args.dataset == 'drive':
        trainset = drive.DRIVEDataset(args.image_dir_train, args.mask_dir_train, transform_train)
        valset = drive.DRIVEDataset(args.image_dir_val, args.mask_dir_val, transform_val)
    
    # dataloader
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    
    # model
    model = Mask2FormerSegmentation(args).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    
    # criterion
    if args.criterion == 'Dice':
        criterion = DiceLoss()
    elif args.criterion == 'Focal':
        criterion = FocalLoss()
    elif args.criterion == 'Tversky':
        criterion = TverskyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # lr_scheduler
    if args.scheduler == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch) # cosine annealing
    else: # constant
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    
    # モデル保存用設定
    min_val_metrics = {
        'loss': 1e+4,
        'accuracy': 0,
        'sensitivity': 0,
        'specificity': 0,
        'iou': 0,
        'dice': 0,
    }
    min_val_epoch = 0
    min_val_model = model.state_dict()
    
    writer = SummaryWriter(log_dir=args.save_dir)
    
    for epoch in tqdm(range(args.max_epoch), ncols=80):
        # 1エポック分の学習
        train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
        scheduler.step()
        # logの書き込み
        with open(train_log_path, 'a') as f:
            f.write(f'{epoch}, {train_loss}' + '\n')
        
        if epoch % args.val_interval == args.val_interval - 1:
            # 検証
            val_loss, acc, sen, spe, iou, dice = evaluate(model, valloader, criterion, epoch, args)
            # tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('Sensitivity', sen, epoch)
            writer.add_scalar('Specificity', spe, epoch)
            writer.add_scalar('IoU', iou, epoch)
            writer.add_scalar('Dice', dice, epoch)
            # csv
            with open(val_log_path, 'a') as f:
                f.write(f'{epoch}, {val_loss}, {acc}, {sen}, {spe}, {iou}, {dice}' + '\n')
            
            # モデル更新
            if val_loss < min_val_metrics['loss']:
                min_val_metrics = {
                    'loss': val_loss,
                    'accuracy': acc,
                    'sensitivity': sen,
                    'specificity': spe,
                    'iou': iou,
                    'dice': dice,
                }
                min_val_epoch = epoch
                min_val_model = model.state_dict()
    
    # モデル保存
    print(f"best epoch: {min_val_epoch}")
    model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(min_val_model, model_path)
    
    writer.add_hparams(hparam_dict=hparams, metric_dict=min_val_metrics)
    writer.close()

def test():
    parser.add_argument('--model_path', required=True, type=str, default='', help='trained model path')
    # parser.add_argument('--double_thresh', type=bool, default=False, help='use double threshold iteration')
    
    args = check_args(mode='test')
    
    # transform
    if args.dataset == 'drive':
        transform_test, num_channels = drive.get_transform(args, mode='test')
    args.num_channels = num_channels
    # dataset
    if args.dataset == 'drive':
        testset = drive.DRIVEDataset(args.image_dir_test, args.mask_dir_test, transform_test)
    
    # dataloader
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    
    # GPU setting
    # model
    model = Mask2FormerSegmentation(args).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint)
    
    acc, sen, spe, iou, dice = test_predict(model, testloader, args)
    print("acc :", acc)
    print("sen :", sen)
    print("spe :", spe)
    print("iou :", iou)
    print("dice :", dice)

if __name__ == '__main__':
    test()