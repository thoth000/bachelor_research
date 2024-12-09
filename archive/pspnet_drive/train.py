import torch
import torch.nn.functional as F

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for sample in dataloader:
        # リサイズされた画像とマスクを使用
        images = sample['transformed_image'].cuda()
        targets = sample['transformed_mask'].cuda()

        optimizer.zero_grad()
        main_out = model(images)
        
        loss = criterion(main_out, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss
