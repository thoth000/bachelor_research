import torch
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import JaccardIndex  # mIoU計算のためのモジュール

from pde.eikonal import levelset_evolution, gradient_sobel
from layers.lse import LSE_output_loss, vector_field_loss, level_map_loss
from dataloaders.class_mapping import get_class_names, get_num_classes
from myconfig import config

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
    torch.cuda.empty_cache()
    
    class_names = get_class_names(dataset)  # データセットに基づいてクラス名を取得
    num_classes = get_num_classes(dataset)  # データセットに基づいてクラス数を取得

    model.eval()
    # iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, average=None, ignore_index=255)
    iou_metric = IoUMetric(num_classes, ignore_index = 255)
    running_loss = 0.0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            torch.cuda.empty_cache()
            
            image = sample['transformed_image'].to(device)
            gts = sample['transformed_mask'].to(device)  # グラウンドトゥルース
            sdt = sample['sdt'].to(device)  # 符号付き距離関数
            dts = sample['dt'].to(device)  # 距離変換
            original_size = sample['mask'].size()[2:]  # 元のマスクサイズを取得

            # モデルからの出力 (phi_0, energy, g)
            phi_0, energy, g = model(image)

            # 出力のサイズを画像に揃える
            target_size = image.size()[2:]
            phi_0 = F.interpolate(phi_0, size=target_size, mode='bilinear', align_corners=True)
            energy = F.interpolate(energy, size=target_size, mode='bilinear', align_corners=True)
            g = F.interpolate(g, size=target_size, mode='bilinear', align_corners=True)

            # ベクトル場をx, yに分割
            batch_size, channels, h, w = energy.size()
            energy = energy.view(batch_size, 2, channels // 2, h, w)

            # 距離変換 (dts) に基づいて勾配データ (vfs) を計算
            vfs = gradient_sobel(dts, split=False)

            # 初期レベルセットの損失を計算 (符号付距離に基づく)
            phi_0_loss = level_map_loss(phi_0, sdt, alpha=config['alpha'])

            # phi_Tのレベルセット進化
            phi_T = levelset_evolution(phi_0, energy, g, T=config['T'], dt_max=config['dt_max'])

            # ベクトル場の損失
            vf_loss = vector_field_loss(energy, vfs, sdt)

            # Tステップ後のレベルセットとグラウンドトゥルースのロス
            phi_T_loss = LSE_output_loss(phi_T, gts, sdt, epsilon=config['epsilon'], dt_max=config['dt_max'])

            # 総損失
            loss = phi_0_loss + phi_T_loss + vf_loss
            running_loss += loss.item()
            
            # print("phi_T resize step")

            # 元のサイズにphi_Tをリサイズ
            phi_T = torch.nn.functional.interpolate(
                phi_T,
                size=original_size,
                mode='bilinear',
                align_corners=True
            )
            
            # print("マスク生成ステップ1")
            
            # リサイズしたphi_Tからマスクを生成
            predicted_mask = (phi_T <= -2).float()
            
            # print("マスク生成ステップ2")
            predicted_mask_ = torch.zeros(predicted_mask.size(0), predicted_mask.size(2), predicted_mask.size(3), device=device)  # [batch_size, height, width]
            for class_id in range(num_classes):
                predicted_mask_[predicted_mask[:, class_id, :, :] == 1] = class_id
            
            del predicted_mask
            
            # print("マスク生成ステップ3")
            
            gts = torch.argmax(sample['mask'], dim=1) # 時間かかりガチ
            
            # print("マスク生成ステップ4")
            
            # ID = 255(無視クラス)を除く
            # iou_metric.update(predicted_mask_.cpu(), gts.cpu())
            # バッチごとのIoUを更新
            iou_metric.update(predicted_mask_.cpu(), gts.cpu())
            
            # print("マスク生成ステップ終了")
            
            # print(predicted_mask_.shape)
            # print(gts.shape)
            # print(valid_mask.shape)

            # マスクを保存する場合
            if save_masks and save_dir:
                img_path = sample['meta']['img_path'][0]
                relative_path = os.path.relpath(img_path, start=os.path.dirname(img_path))

                epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}' if epoch is not None else 'inference')
                save_path = os.path.join(epoch_dir, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                mask_np = predicted_mask_.squeeze(0).cpu().numpy().astype(np.uint8)
                mask_file = save_path.replace('_leftImg8bit.png', '_pred_mask.png')
                save_colored_mask(mask_np, mask_file, num_classes)

    avg_loss = running_loss / len(dataloader)
    # classwise_iou = iou_metric.compute()  # 各クラスごとのIoU
    # miou = classwise_iou.mean().item()    # 全クラスの平均IoU（mIoU）
    # 全体のIoUとmIoUを計算
    miou, classwise_iou = iou_metric.compute()

    return avg_loss, miou, classwise_iou


def save_colored_mask(mask, save_path, num_classes):
    """
    予測されたマスクをカラーマップを使って保存します。
    :param mask: 予測されたマスク (numpy array)
    :param save_path: 保存するファイルパス
    :param num_classes: クラスの総数
    """
    cmap = plt.get_cmap('tab20', num_classes)
    colored_mask = cmap(mask / (num_classes - 1))
    plt.imsave(save_path, colored_mask)
