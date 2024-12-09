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
    # 0ステップ後のレベルセット関数を対象
    total_tp_0 = torch.tensor(0.0, device=device)
    total_tn_0 = torch.tensor(0.0, device=device)
    total_fp_0 = torch.tensor(0.0, device=device)
    total_fn_0 = torch.tensor(0.0, device=device)
    # 1ステップ後のレベルセット関数を対象
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
            
            phi_0, P = model(images)
        
            # 初期レベルセットのロス計算
            if epoch < args.pre_epoch or True:
                loss_0 = level_map_loss(phi_0, sample['sdt'].to(device), alpha=1)
                loss_P = P_loss(P, sample['P'].to(device))
                phi_T = levelset_evolution_pde(phi_0, images, P)
                loss_T = LSE_output_loss(phi_T, sample['transformed_mask'].to(device))
                loss = loss_0 + loss_P + loss_T
            else:
                phi_T = levelset_evolution(phi_0, images, P)
                loss_T = LSE_output_loss(phi_T, sample['transformed_mask'].to(device))
                # print('phi_T:', loss_T.item())
                loss = loss_T
            
            running_loss += loss * images.size(0) # バッチ内のサンプル数で加重
            total_samples += images.size(0)  # バッチ内のサンプル数を加算
            
            # 評価
            masks = sample['mask'].to(device) # 元サイズのマスク
            mask_size = masks.shape[2:]
            phi_0_resized = F.interpolate(phi_0, size=mask_size, mode='bilinear')
            phi_T_resized = F.interpolate(phi_T, size=mask_size, mode='bilinear')
            if args.rank == 0 and (args.save_mask or epoch == args.max_epoch - 1) and i < 2:
                # save_main_out_image(sample['sdt'], os.path.join(out_dir, f"{i}_sdt.png"))
                save_main_out_image(phi_0_resized, os.path.join(out_dir, f"{i}_phi_0.png"))
                save_main_out_image(phi_T_resized, os.path.join(out_dir, f"{i}_phi_T.png"))
                save_main_out_image(sample['P'], os.path.join(out_dir, f"{i}_P_gt.png"))
                save_main_out_image(sample['Rb'], os.path.join(out_dir, f"{i}_Rb.png"))
                save_main_out_image(sample['S'], os.path.join(out_dir, f"{i}_S.png"))
                save_main_out_image(sample['centerline'], os.path.join(out_dir, f"{i}_centerline.png"))
                save_main_out_image(P, os.path.join(out_dir, f"{i}_P.png"))
            
            preds_0 = (phi_0_resized <= args.threshold).float()
            preds_T = (phi_T_resized <= args.threshold).float()
            # 元画像サイズにする
            preds_0 = unpad_to_original(preds_0, sample["padding"])
            preds_T = unpad_to_original(preds_T, sample["padding"])
            masks = unpad_to_original(masks, sample["padding"])
            
            # TP, TN, FP, FNの計算(0ステップ)
            total_tp_0 += torch.sum((preds_0 == 1) & (masks == 1)).item()
            total_tn_0 += torch.sum((preds_0 == 0) & (masks == 0)).item()
            total_fp_0 += torch.sum((preds_0 == 1) & (masks == 0)).item()
            total_fn_0 += torch.sum((preds_0 == 0) & (masks == 1)).item()
            
            # TP, TN, FP, FNの計算(Tステップ)
            total_tp += torch.sum((preds_T == 1) & (masks == 1)).item()
            total_tn += torch.sum((preds_T == 0) & (masks == 0)).item()
            total_fp += torch.sum((preds_T == 1) & (masks == 0)).item()
            total_fn += torch.sum((preds_T == 0) & (masks == 1)).item()
    
    # 集計
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    dist.all_reduce(total_tp_0, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn_0, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp_0, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn_0, op=dist.ReduceOp.SUM)
    
    dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)
    
    avg_loss = running_loss / total_samples
    
    acc_0 = (total_tp_0 + total_tn_0) / (total_tp_0 + total_tn_0 + total_fp_0 + total_fn_0)
    sen_0 = total_tp_0 / (total_tp_0 + total_fn_0) if (total_tp_0 + total_fn_0) > 0 else torch.tensor(0.0, device=device)
    spe_0 = total_tn_0 / (total_tn_0 + total_fp_0) if (total_tn_0 + total_fp_0) > 0 else torch.tensor(0.0, device=device)
    iou_0 = total_tp_0 / (total_tp_0 + total_fp_0 + total_fn_0) if (total_tp_0 + total_fp_0 + total_fn_0) > 0 else torch.tensor(0.0, device=device)
    miou_0 = ((total_tp_0 / (total_tp_0 + total_fp_0 + total_fn_0)) + (total_tn_0 / (total_tn_0 + total_fp_0 + total_fn_0))) / 2
    dice_0 = (2 * total_tp_0) / (2 * total_tp_0 + total_fp_0 + total_fn_0) if (2 * total_tp_0 + total_fp_0 + total_fn_0) > 0 else torch.tensor(0.0, device=device)

    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sen = total_tp / (total_tp + total_fn)
    spe = total_tn / (total_tn + total_fp)
    iou = total_tp / (total_tp + total_fp + total_fn)
    miou = ((total_tp / (total_tp + total_fp + total_fn)) + (total_tn / (total_tn + total_fp + total_fn))) / 2
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) # F1
    
    return avg_loss.item(), acc_0.item(), sen_0.item(), spe_0.item(), iou_0.item(), miou_0.item(), dice_0.item(), acc.item(), sen.item(), spe.item(), iou.item(), miou.item(), dice.item()


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