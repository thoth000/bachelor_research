import torch
import torch.nn.functional as F
import torch.distributed as dist

def train_one_epoch(model, dataloader, criterion, optimizer, device, world_size):
    model.train()
    train_loss = torch.tensor(0.0, device=device)  # train_lossをテンソルで初期化

    for i, sample in enumerate(dataloader):
        images = sample['transformed_image'].to(device)
        targets = sample['transformed_mask'].to(device)

        optimizer.zero_grad()
        main_out = model(images)
        
        loss = criterion(main_out, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss  # 累積

    # DDP: 全プロセスで損失を集約
    if dist.is_initialized():
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss /= world_size  # プロセス数で割って平均化

    avg_loss = train_loss.item() / len(dataloader)  # 最終的にスカラー化してからバッチ数で平均
    return avg_loss