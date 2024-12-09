import os
import torch
import csv
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from delse_psp import DELSE_PSP
from train import train_one_epoch
from evaluate import evaluate
import dataloaders.cityscapes_loader as cityscapes
from dataloaders.class_mapping import get_class_names, get_num_classes
from myconfig import config

class_names = get_class_names(config['dataset'])

def save_config(run_dir):
    config_file = os.path.join(run_dir, "config.txt")
    with open(config_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def write_iou_to_csv(csv_file, epoch, loss, miou, classwise_iou):
    fieldnames = ['Epoch', 'Loss', 'mIoU'] + class_names
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {
            'Epoch': epoch,
            'Loss': loss,  # 統一されたフィールド名
            'mIoU': miou
        }
        row.update({class_name: iou.item() for class_name, iou in zip(class_names, classwise_iou)})
        writer.writerow(row)

def train():
    run_id = len(os.listdir('exp'))
    run_dir = os.path.join('exp', f'run_{run_id:03d}')
    os.makedirs(run_dir, exist_ok=True)
    
    save_config(run_dir)

    model = DELSE_PSP(num_classes=get_num_classes(config["dataset"]))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if config["model_path"] is not None:
        print("Trained model is loaded.")
        model.load_state_dict(torch.load(config["model_path"]))
    model = model.to(config["device"])

    train_log_file = os.path.join(run_dir, 'train_log.csv')
    val_log_file = os.path.join(run_dir, 'val_log.csv')

    with open(train_log_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss'])

    with open(val_log_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss', 'mIoU'] + class_names)

    # Set optimizer
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config["factor"], patience=config["patience"], threshold=config["threshold"], cooldown=config["cooldown"], min_lr=config["min_lr"])

    train_loader = cityscapes.get_dataloader(split='train')
    val_loader = cityscapes.get_dataloader(split='val')

    # Print training information
    print(f"Starting training with the following configuration:")
    print(f"  Run directory: {run_dir}")
    print(f"  Dataset : {config['dataset']}")
    print(f"  Total epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Initial learning rate: {config['lr']}")
    print(f"  Optimizer: {config['optimizer']}")

    for epoch in tqdm(range(config["epochs"])):
        train_loss = train_one_epoch(model, train_loader, optimizer, config["device"])
        with open(train_log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss])
        
        val_loss, miou, classwise_iou = evaluate(model, val_loader, config["device"], save_masks=False, save_dir=run_dir, epoch=epoch, dataset=config['dataset'])
        write_iou_to_csv(val_log_file, epoch+1, val_loss, miou, classwise_iou)
        
        scheduler.step(val_loss)

    model_save_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(model.state_dict(), model_save_path)

def test():
    model = DELSE_PSP(num_classes=get_num_classes(config["dataset"]))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(config["model_path"]))
    model = model.to(config["device"])

    # ディレクトリが存在しない場合に作成する
    os.makedirs(config["output_dir"], exist_ok=True)

    test_loader = cityscapes.get_dataloader(split='test')
    test_log_file = os.path.join(config["output_dir"], 'test_log.csv')
    test_loss, miou, classwise_iou = evaluate(model, test_loader, config["device"], dataset=config['dataset'], save_masks=True, save_dir=config["output_dir"])
    write_iou_to_csv(test_log_file, 0, test_loss, miou, classwise_iou)

if __name__ == '__main__':
    if config["mode"] == 'train':
        train()
    elif config["mode"] == 'test':
        test()
