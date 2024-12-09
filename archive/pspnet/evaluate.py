import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import JaccardIndex  # mIoU計算のためのモジュール

# クラス名とラベルIDのマッピングを読み込む
from dataloaders.class_mapping import get_class_names, get_num_classes

def compute_loss(main_out, aux_out, target, criterion, aux_weight=0.4):
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

class IoUMetric:
    def __init__(self, num_classes, ignore_index=255):
        """
        IoUの計算と更新を行うクラス
        :param num_classes: クラス数
        :param ignore_index: 無視するクラスID
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.total_ious = torch.zeros(num_classes, 2)  # [クラス数, 交差部分（intersection）, 和集合部分（union）]

    def update(self, pred_mask, true_mask):
        """
        バッチごとのIoU情報を更新
        :param pred_mask: モデルの予測マスク [batch_size, H, W] (予測されたクラスID)
        :param true_mask: 実際のマスク [batch_size, H, W] (グラウンドトゥルースクラスID)
        """
        valid_mask = (true_mask != self.ignore_index)

        for class_id in range(self.num_classes):
            pred_class = (pred_mask == class_id) & valid_mask
            true_class = (true_mask == class_id) & valid_mask

            intersection = (pred_class & true_class).sum().item()
            union = (pred_class | true_class).sum().item()

            self.total_ious[class_id, 0] += intersection  # intersection を更新
            self.total_ious[class_id, 1] += union  # union を更新

    def compute(self):
        """
        蓄積されたIoU情報からmIoUとクラスごとのIoUを計算
        :return: mIoU（平均IoU）とクラスごとのIoU
        """
        class_iou = self.total_ious[:, 0] / (self.total_ious[:, 1] + 1e-6)  # IoU = intersection / union
        miou = torch.nanmean(class_iou).item()  # NaNを無視して平均IoU (mIoU) を計算
        return miou, class_iou

    def reset(self):
        """
        IoUの蓄積をリセット
        """
        self.total_ious = torch.zeros(self.num_classes, 2)

def evaluate(model, dataloader, device, save_masks=False, save_dir=None, epoch=None, dataset='cityscapes'):
    class_names = get_class_names(dataset)  # データセットに基づいてクラス名を取得
    num_classes = get_num_classes(dataset)  # データセットに基づいてクラス数を取得

    model.eval()
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=255)
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            transformed_image = sample['transformed_image'].to(device)
            transformed_mask = sample['transformed_mask'].to(device)
            original_size = sample['mask'].size()[1:]

            main_out, aux_out = model(transformed_image)
            loss = compute_loss(main_out, aux_out, transformed_mask, criterion)
            # loss = criterion(outputs, transformed_mask)
            running_loss += loss.item()

            outputs = torch.nn.functional.interpolate(
                main_out,  # ここは連続値 (logits)
                size=original_size,  # 元の画像サイズに合わせる
                mode='bilinear',  # bilinear 補間を使用
                align_corners=False
            )
            # Step 2: クラスIDを取得 (補間後にargmaxを適用)
            outputs = torch.argmax(outputs, dim=1)  # 各ピクセルで最も高いスコアのクラスIDを選択

            iou_metric.update(outputs.cpu(), sample['mask'].cpu())

            if save_masks and save_dir:
                img_path = sample['meta']['img_path'][0]
                relative_path = os.path.relpath(img_path, start=os.path.dirname(img_path))

                epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}' if epoch is not None else 'inference')
                save_path = os.path.join(epoch_dir, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                mask = outputs.squeeze(0).cpu().numpy().astype(np.uint8)
                mask_file = save_path.replace('_leftImg8bit.png', '_pred_mask.png')
                save_colored_mask(mask, mask_file, num_classes)

    avg_loss = running_loss / len(dataloader)
    miou, classwise_iou = iou_metric.compute()  # 各クラスごとのIoU
    # miou = classwise_iou.mean().item()

    return avg_loss, miou, classwise_iou

def save_colored_mask(mask, save_path, num_classes):
    cmap = plt.get_cmap('tab20', num_classes)
    colored_mask = cmap(mask / (num_classes - 1))
    plt.imsave(save_path, colored_mask)
