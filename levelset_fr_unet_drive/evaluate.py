import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt

from dataloader.drive_loader import unpad_to_original

from models.lse import *
from models.loss import *

# loss, accuracy, IoU, Dice
def evaluate(model, dataloader, epoch, args, device):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    
    # TP, TN, FP, FN
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    if args.save_mask or epoch == args.max_epoch - 1:
        # ディレクトリ生成
        out_dir= os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['transformed_image'].to(device)
            
            z = model(images)
            y = torch.softmax(z, dim=1)
            # 初期レベルセットのロス計算
            if epoch < args.pre_epoch:
                y1 = y[:, 1:2, :, :]
                gt = sample['transformed_mask'].to(device)
                loss = selected_loss(args, y1, gt, z, y, mode="pre_loss")
            else:
                y1 = y[:, 1:2, :, :]
                gt = sample['transformed_mask'].to(device)
                loss = selected_loss(args, y1, gt, z, y, mode="post_loss")
            
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            mask_size = masks.shape[2:]
            y1 = F.interpolate(y1, size=mask_size, mode='bilinear')
            
            preds = (y1 > args.threshold).float()
            # 元画像サイズにする
            preds = unpad_to_original(preds, sample["padding"])
            masks = unpad_to_original(masks, sample["padding"])
            
            if args.save_mask and args.rank == 0:
                # マスクを保存
                save_main_out_image(y1, os.path.join(out_dir, f"mask_{i}_y1.png"))
            
            # TP, TN, FP, FNの計算
            total_tp += torch.sum((preds == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds == 0) & (masks == 1)).item()
    
    # 集計
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    
    avg_loss = running_loss / total_samples
    
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn)
    spe = total_tn / (total_tn + total_fp)
    iou = total_tp / (total_tp + total_fp + total_fn)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) # F1
    
    return avg_loss.item(), acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item()


def save_main_out_image(output_tensor, filepath, cmap="viridis"):
    """
    output_tensorをカラーマップで画像として保存する関数
    :param output_tensor: 保存するテンソル
    :param filepath: 画像の保存先のパス
    :param cmap: 使用するカラーマップ（デフォルトは 'viridis'）
    """
    # output_tensorをnumpy配列に変換
    output_numpy = output_tensor.squeeze().cpu().detach().numpy()

    # カラーマップを使って画像として保存
    plt.imshow(output_numpy, cmap=cmap)
    plt.colorbar()
    plt.savefig(filepath)  # ファイルパスに画像を保存
    plt.close()  # メモリを解放するためにプロットを閉じる


def save_vector_field(sdt, vx, vy, filepath, step=100, scale=10, cmap="viridis"):
    """
    SDTをカラーマップで表示し、ベクトル場を白い矢印で重ねて保存する関数。
    
    :param sdt: SDT（Signed Distance Transform）を表す2次元配列
    :param vx: x方向のベクトル成分
    :param vy: y方向のベクトル成分
    :param filepath: 保存する画像のパス
    :param step: 矢印を描画する間隔
    :param scale: 矢印のスケール
    :param cmap: SDTのカラーマップ
    """
    # 入力をnumpy配列に変換
    sdt_numpy = sdt.squeeze().cpu().detach().numpy()
    vx_numpy = vx.squeeze().cpu().detach().numpy()
    vy_numpy = vy.squeeze().cpu().detach().numpy()

    # メッシュグリッドを生成
    H, W = sdt_numpy.shape
    X, Y = torch.meshgrid(
        torch.arange(W),  # 横方向
        torch.arange(H),  # 縦方向
        indexing="xy"
    )

    # 矢印の間隔を空ける
    X = X.numpy()[::step, ::step]
    Y = Y.numpy()[::step, ::step]
    vx_numpy = vx_numpy[::step, ::step]
    vy_numpy = vy_numpy[::step, ::step]
    
    # 正規化
    # norm = np.sqrt(vx_numpy**2 + vy_numpy**2 + 1e-10)
    # vx_numpy_reg = vx_numpy / norm
    # vy_numpy_reg = vy_numpy / norm
    
    # print('normal:', np.max(vx_numpy**2 + vy_numpy**2), 'reg:', np.max(vx_numpy_reg**2 + vy_numpy_reg**2))

    # 描画
    plt.figure(figsize=(8, 8))
    
    # SDTをカラーマップで表示
    plt.imshow(sdt_numpy, cmap=cmap, origin="upper")
    plt.colorbar(label="SDT Value")
    
    # ベクトル場を白い矢印で重ねる
    plt.quiver(X, Y, vx_numpy, vy_numpy, color="white", scale=scale)

    # 軸ラベルとタイトル
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("SDT with Vector Field")
    
    # 保存
    plt.savefig(filepath)
    plt.close()


def save_img(img, filepath):
    """
    画像を保存する関数
    :param img: 保存する画像
    :param filepath: 画像の保存先のパス
    """
    # 画像をnumpy配列に変換
    img_numpy = img.squeeze().cpu().detach().numpy()

    # 画像を保存
    plt.imshow(img_numpy, cmap="gray")
    plt.axis("off")
    plt.savefig(filepath)  # ファイルパスに画像を保存
    plt.close()  # メモリを解放するためにプロットを閉じる