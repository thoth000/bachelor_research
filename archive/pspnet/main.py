import os
import torch
import csv
from torch import optim
from tqdm import tqdm

from pspnet import PSPNet
from train import train_one_epoch
from evaluate import evaluate
from dataloaders.cityscapes_loader import get_dataloader
from myconfig import config

def save_config(run_dir):
    config_file = os.path.join(run_dir, "config.txt")
    with open(config_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def write_iou_to_csv(csv_file, epoch, loss, miou, classwise_iou):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a') as f:
        writer = csv.writer(f)

        row = [epoch, loss, miou]
        row.extend(iou.item() for iou in classwise_iou)
        writer.writerow(row)

def train():
    run_id = len(os.listdir('exp'))
    run_dir = os.path.join('exp', f'run_{run_id:03d}')
    os.makedirs(run_dir, exist_ok=True)
    
    save_config(run_dir)

    model = PSPNet(num_classes=config["num_classes"])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(config["device"])

    train_log_file = os.path.join(run_dir, 'train_log.csv')
    val_log_file = os.path.join(run_dir, 'val_log.csv')

    with open(train_log_file, 'w') as f:
        pass  # ラベルなしでログを開始

    with open(val_log_file, 'w') as f:
        pass  # ラベルなしでログを開始

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
        
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    train_loader = get_dataloader(split='train')
    val_loader = get_dataloader(split='val')

    # Print training information
    print(f"Starting training with the following configuration:")
    print(f"  Run directory: {run_dir}")
    print(f"  Total epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Optimizer: {config['optimizer']}")

    for epoch in tqdm(range(config["epochs"])):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config["device"])

        with open(train_log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss])

        if (epoch + 1) % config["eval_interval"] == 0:
            val_loss, miou, classwise_iou = evaluate(model, val_loader, config["device"], save_masks=False, save_dir=run_dir, epoch=epoch, dataset=config['dataset'])
            write_iou_to_csv(val_log_file, epoch+1, val_loss, miou, classwise_iou)

    model_save_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(model.state_dict(), model_save_path)

def test():
    model = PSPNet(num_classes=config["num_classes"])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(config["model_path"]))
    model = model.to(config["device"])

    test_loader = get_dataloader(split='test')
    test_log_file = os.path.join(config["output_dir"], 'test_log.csv')
    test_loss, miou, classwise_iou = evaluate(model, test_loader, config["device"], dataset=config['dataset'], save_masks=True, save_dir=config["output_dir"])
    write_iou_to_csv(test_log_file, 0, test_loss, miou, classwise_iou)

if __name__ == '__main__':
    if config["mode"] == 'train':
        train()
    elif config["mode"] == 'test':
        test()
        print("finish")
