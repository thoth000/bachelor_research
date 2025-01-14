import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

import dataloader.drive_loader as drive


def minpool(input, kernel_size=3, stride=1, padding=1):
    """
    最小プーリング
    Args:
        input (torch.Tensor): 入力テンソル (N, C, H, W)
        kernel_size (int): カーネルサイズ
        stride (int): ストライド
        padding (int): パディング
    Returns:
        torch.Tensor: 出力テンソル (N, C, H, W)
    """
    return F.max_pool2d(input*-1, kernel_size, stride, padding)*-1 # 最大プーリングを適用して再度反転


def soft_skeleton(mask, k=30):
    """
    ソフトスケルトン変換
    Args:
        mask (torch.Tensor): マスク画像 (N, 1, H, W)
        k (int): 最大管幅
    Returns:
        torch.Tensor: ソフトスケルトン画像 (N, 1, H, W)
    """
    # Initialize I' as maxpool(minpool(mask))
    I_prime = F.max_pool2d(minpool(mask, kernel_size=3, stride=1, padding=1), kernel_size=3, stride=1, padding=1)
    # Initialize S as ReLU(I - I')
    S = F.relu(mask - I_prime)

    # Iterative refinement of the skeleton
    for _ in range(k):
        # Update I
        mask = minpool(mask, kernel_size=3, stride=1, padding=1)
        # Update I'
        I_prime = F.max_pool2d(minpool(mask, kernel_size=3, stride=1, padding=1), kernel_size=3, stride=1, padding=1)
        # Update S
        S = S + (1 - S) * F.relu(mask - I_prime)

    return S


def skeletonize_mask(mask):
    """
    スケルトン化のみを行う関数
    Args:
        mask (numpy.ndarray): 入力マスク (H, W) バイナリマスク
    Returns:
        skeleton (numpy.ndarray): スケルトン (H, W) バイナリマスク
    """
    binary_mask = (mask > 0).astype(np.uint8)  # 0, 1 のみのバイナリマスク
    return skeletonize(binary_mask)


def calculate_skeleton_directions_with_pca(skeleton, window_size=5):
    """
    スケルトン上の方向ベクトルを主成分分析（PCA）で計算。
    Args:
        skeleton (numpy.ndarray): スケルトン (H, W) バイナリマスク
        window_size (int): 方向を計算する際に考慮する近傍のサイズ（奇数）
    Returns:
        directions (numpy.ndarray): スケルトン点ごとの方向ベクトル (H, W, 2)
    """
    H, W = skeleton.shape
    directions = np.zeros((H, W, 2))  # 初期化
    half_window = window_size // 2

    x, y = np.nonzero(skeleton)  # スケルトン点を取得

    for xi, yi in zip(x, y):
        # 近傍範囲を計算
        x_start, x_end = max(xi - half_window, 0), min(xi + half_window + 1, H)
        y_start, y_end = max(yi - half_window, 0), min(yi + half_window + 1, W)

        # 近傍のスケルトン点を取得
        neighbors = skeleton[x_start:x_end, y_start:y_end]
        neighbor_x, neighbor_y = np.nonzero(neighbors)

        # 近傍内の座標をPCAに渡す
        coords = np.stack([neighbor_x + x_start, neighbor_y + y_start], axis=1)
        if coords.shape[0] > 1:  # PCAを実行するには2点以上が必要
            pca = PCA(n_components=2)
            pca.fit(coords)
            principal_direction = pca.components_[0]  # 第2主成分が主方向
            if principal_direction[0] < 0:  # x方向が負なら反転
                directions[xi, yi] = -principal_direction
            else:
                directions[xi, yi] = principal_direction

    return directions


def calculate_skeleton_directions_with_central_difference(skeleton):
    """
    スケルトン上の方向ベクトルを中心差分法で計算。
    Args:
        skeleton (numpy.ndarray): スケルトン (H, W) バイナリマスク
    Returns:
        directions (numpy.ndarray): スケルトン点ごとの方向ベクトル (H, W, 2)
    """
    H, W = skeleton.shape
    directions = np.zeros((H, W, 2))
    skeleton = skeleton.astype(np.int8)  # 整数型に変換

    x, y = np.nonzero(skeleton)  # スケルトン点を取得

    for xi, yi in zip(x, y):
        # 近傍の差分を計算
        dx = skeleton[max(xi-1, 0):min(xi+2, H), yi] - skeleton[max(xi-2, 0):min(xi+1, H), yi]
        dy = skeleton[xi, max(yi-1, 0):min(yi+2, W)] - skeleton[xi, max(yi-2, 0):min(yi+1, W)]

        # 差分ベクトルを計算
        direction = np.array([np.sum(dx), np.sum(dy)])

        if np.linalg.norm(direction) > 0:  # ゼロベクトルでない場合
            direction = direction / np.linalg.norm(direction)
            if direction[0] < 0:  # x方向が負なら反転
                directions[xi, yi] = -direction
            else:
                directions[xi, yi] = direction

    return directions


def calculate_skeleton_directions(skeleton, initial_window_size=5, max_window_size=50):
    """
    スケルトン上の各点の方向を計算し、x方向が正になるよう制限。
    ゼロベクトルが発生した場合、ウィンドウサイズを動的に広げて再計算。

    Args:
        skeleton (numpy.ndarray): スケルトン (H, W) バイナリマスク
        initial_window_size (int): 最初の近傍ウィンドウサイズ（奇数）
        max_window_size (int): 最大の近傍ウィンドウサイズ（奇数）

    Returns:
        directions (numpy.ndarray): スケルトン点ごとの方向ベクトル (H, W, 2)
    """
    H, W = skeleton.shape
    directions = np.zeros((H, W, 2))  # 初期化

    x, y = np.nonzero(skeleton)  # スケルトンの点を取得

    for xi, yi in zip(x, y):
        window_size = initial_window_size
        direction_found = False

        while not direction_found and window_size <= max_window_size:
            half_window = window_size // 2
            # 近傍範囲を計算
            x_start, x_end = max(xi - half_window, 0), min(xi + half_window + 1, H)
            y_start, y_end = max(yi - half_window, 0), min(yi + half_window + 1, W)

            # 近傍のスケルトン点を取得
            neighbors = skeleton[x_start:x_end, y_start:y_end]
            neighbor_x, neighbor_y = np.nonzero(neighbors)
            
            # 近傍内の相対座標を計算
            directions_local = np.stack(
                [neighbor_x - (xi - x_start), neighbor_y - (yi - y_start)],
                axis=1,
            )
            mean_direction = np.mean(directions_local, axis=0)

            if np.linalg.norm(mean_direction) > 0:  # ゼロベクトルでない場合
                direction = mean_direction / np.linalg.norm(mean_direction)
                if direction[0] < 0:  # x方向が負なら反転
                    directions[xi, yi] = -direction
                else:
                    directions[xi, yi] = direction
                direction_found = True  # 有効な方向が見つかった場合
            else:
                window_size += 2  # ウィンドウサイズを広げる

        if not direction_found:
            # 最大ウィンドウサイズでも方向が見つからなかった場合
            directions[xi, yi] = [0, 0]  # ゼロベクトルを割り当て

    return directions


def find_nearest_skeleton_directions(vessel_mask, skeleton, skeleton_directions):
    """
    血管上の点に最も近いスケルトンの方向を割り当て
    """
    distance, indices = distance_transform_edt(~skeleton, return_indices=True)

    vessel_directions = np.zeros((*vessel_mask.shape, 2))
    x, y = np.nonzero(vessel_mask)

    for xi, yi in zip(x, y):
        nearest_x, nearest_y = indices[:, xi, yi]
        if skeleton[nearest_x, nearest_y]:  # 最近接点がスケルトン上であることを確認
            direction = skeleton_directions[nearest_x, nearest_y]
            vessel_directions[xi, yi] = direction

    return vessel_directions


def visualize_skeleton_directions(skeleton, skeleton_directions, file_path, scale=0.1):
    """
    スケルトン上の方向ベクトルを可視化して保存
    """
    u = skeleton_directions[..., 0]
    v = skeleton_directions[..., 1]

    y, x = np.nonzero(skeleton)
    u_sampled = u[y, x]
    v_sampled = v[y, x]

    plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap="gray", interpolation="nearest", alpha=0.6)
    plt.quiver(x, y, u_sampled, v_sampled, angles='xy', scale_units='xy', scale=scale, color='red')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_vector_field_with_skeleton(vector_field, skeleton, file_path, scale=0.1):
    """
    ベクトル場をスケルトンを背景に可視化して保存
    """
    u = vector_field[..., 0]
    v = vector_field[..., 1]
    H, W = u.shape

    # u, vが0でない点のみをサンプリング
    y,x = np.nonzero((u != 0) | (v != 0))
    u_sampled = u[y, x]
    v_sampled = v[y, x]

    plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap="gray", interpolation="nearest", alpha=0.6)
    plt.quiver(x, y, u_sampled, v_sampled, angles='xy', scale_units='xy', scale=scale, color='blue')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def count_nonzero_vectors(directions):
    """
    非ゼロベクトルの数をカウント
    Args:
        directions (numpy.ndarray): ベクトル場 (H, W, 2)
    Returns:
        int: 非ゼロベクトルの数
    """
    norms = np.linalg.norm(directions, axis=2)  # 各ベクトルのノルムを計算
    return np.count_nonzero(norms)  # ノルムが0でないベクトルをカウント


def count_nonzero_scalars(scalar_field):
    """
    スカラー場の非ゼロ要素をカウント
    Args:
        scalar_field (numpy.ndarray): スカラー場 (H, W)
    Returns:
        int: 非ゼロ要素の数
    """
    return np.count_nonzero(scalar_field)  # 0でない要素をカウント



if __name__ == "__main__":
    transform = drive.get_transform(None, mode='test')
    dataset = drive.DRIVEDataset(mode="test", path="/home/sano/dataset/DRIVE", opt="pad", transform=transform)

    sample = dataset[0]
    output_dir = "output"
    
    mask = sample['mask'].cpu().numpy().transpose(1, 2, 0).squeeze()
    skeleton = skeletonize_mask(mask) # スケルトン化
    # skeleton_directions = calculate_skeleton_directions(skeleton) # 方向ベクトル計算
    # skeleton_directions = calculate_skeleton_directions_with_pca(skeleton) # PCAで方向ベクトル計算
    skeleton_directions = calculate_skeleton_directions_with_central_difference(skeleton) # 中心差分法で方向ベクトル計算
    vessel_directions = find_nearest_skeleton_directions(mask, skeleton, skeleton_directions) # 血管上の点に最も近いスケルトンの方向を割り当て

    # 結果を可視化
    visualize_skeleton_directions(skeleton, skeleton_directions, os.path.join(output_dir, "skeleton_directions.png"))
    visualize_vector_field_with_skeleton(vessel_directions, skeleton, os.path.join(output_dir, "vessel_directions_with_skeleton.png"))

    # skeleton の非ゼロスカラー数を出力
    nonzero_skeleton = count_nonzero_scalars(skeleton)
    print(f"Non-zero skeleton scalars: {nonzero_skeleton}")
    
    # mask の非ゼロスカラー数を出力
    nonzero_mask = count_nonzero_scalars(mask)
    print(f"Non-zero mask scalars: {nonzero_mask}")

    nonzero_skeleton = count_nonzero_vectors(skeleton_directions)
    print(f"Non-zero skeleton directions: {nonzero_skeleton}")

    # vessel_directions の非ゼロベクトル数を出力
    nonzero_vessel = count_nonzero_vectors(vessel_directions)
    print(f"Non-zero vessel directions: {nonzero_vessel}")
