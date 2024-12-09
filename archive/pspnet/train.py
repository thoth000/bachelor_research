import torch
import torch.nn.functional as F

def compute_loss(main_out, aux_out, target, criterion, aux_weight=0.2):
    target_size = target.size()[1:]  # [H, W] size of the target mask

    # Resize auxiliary output to match target size
    aux_out = F.interpolate(aux_out, size=target_size, mode='bilinear', align_corners=False)

    # Compute the main loss
    main_loss = criterion(main_out, target)

    # Compute the auxiliary loss
    aux_loss = criterion(aux_out, target)

    # Total loss = main loss + aux_weight * auxiliary loss
    total_loss = main_loss + aux_weight * aux_loss

    return total_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for sample in dataloader:
        # リサイズされた画像とマスクを使用
        images = sample['transformed_image'].to(device)
        targets = sample['transformed_mask'].to(device)

        optimizer.zero_grad()
        main_out, aux_out = model(images)
        
        loss = compute_loss(main_out, aux_out, targets, criterion)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss
