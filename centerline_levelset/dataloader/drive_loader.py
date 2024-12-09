import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, gaussian_filter
import cv2
from PIL import Image
import numpy as np

import albumentations as A

class PadToSquare:
    """画像とマスクを正方形のnumpy配列にパディングするクラス"""
    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]

        max_side = max(h, w)
        
        # パディングの量を計算して保存
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left

        sample['padding'] = (pad_top, pad_bottom, pad_left, pad_right)
        
        # 画像の次元数に応じたパディングの適用
        if image.ndim == 3:  # カラー画像の場合
            padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=self.fill_value)
        else:  # グレースケール画像の場合
            padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=self.fill_value)
        
        # マスクの次元数に応じたパディングの適用
        if mask.ndim == 3:
            padded_mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=self.fill_value)
        else:
            padded_mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=self.fill_value)
        
        sample['image'] = padded_image
        sample['mask'] = padded_mask
        
        return sample

def unpad_to_original(output, padding):
    # ネットワークの出力からパディングを解除し、元のサイズに戻す関数。
    pad_top, pad_bottom, pad_left, pad_right = padding

    if isinstance(output, torch.Tensor):
        # テンソルの場合
        unpadded_output = output[:, pad_top:output.shape[1]-pad_bottom, pad_left:output.shape[2]-pad_right]
    else:
        # NumPy 配列の場合
        unpadded_output = output[pad_top:output.shape[0]-pad_bottom, pad_left:output.shape[1]-pad_right]
    
    return unpadded_output


class StandardTransform:
    """すべての変換をまとめたクラス"""
    def __init__(self):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
        ])

    def __call__(self, sample):
        # Albumentations での変換を適用
        augmented = self.transforms(image=sample['image'], mask=sample['mask'])
        sample['image'] = augmented['image']
        sample['mask'] = augmented['mask']

        return sample


class Resize:
    """画像とマスクをnumpy配列でリサイズするクラス"""
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        sample['transformed_image'] = cv2.resize(sample['image'], self.size, interpolation=cv2.INTER_LINEAR)
        sample['transformed_mask'] = cv2.resize(sample['mask'], self.size, interpolation=cv2.INTER_NEAREST)
        return sample


class Keep:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['transformed_image'] = sample['image']
        sample['transformed_mask'] = sample['mask']
        return sample


class ComputeSignedDistance: # 負: 前景, 正: 背景
    """1チャネルのマスクから符号付き距離マップと距離変換を計算し、[batch, 1, H, W]の形で保存するクラス"""
    def __init__(self, max_distance=30):
        self.max_distance = max_distance

    def __call__(self, sample):
        transformed_mask = sample['transformed_mask']
        
        # 前景と背景の距離変換を計算
        # distance_transform_edt : 最も近い 0 までのユークリッド距離を計算する
        dist_in = distance_transform_edt(transformed_mask) # 背景からの距離変換
        dist_out = distance_transform_edt(1 - transformed_mask) # 前景からの距離変換
        
        # 符号付き距離の計算
        signed_distance_map = dist_out - dist_in
        signed_distance_map = np.clip(signed_distance_map, -self.max_distance, self.max_distance)
        
        # バッチ次元とチャネル次元を追加して [batch, 1, H, W] の形に変換
        sample['sdt'] = signed_distance_map
        sample['dt'] = dist_in + dist_out

        return sample

class VesselnessFilter:
    def __init__(self, scales, beta=2.0, c=5.0, a=0.5, b=0.2):
        self.scales = scales
        self.beta = beta
        self.c = c
        self.a = a
        self.b = b

    @staticmethod
    def apply_gaussian_smoothing(image, sigma):
        """
        Applies Gaussian smoothing.

        Args:
            image (numpy.ndarray): Input image (H, W) or (H, W, 1).
            sigma (float): Standard deviation for Gaussian kernel.

        Returns:
            numpy.ndarray: Smoothed image with the same shape as input.
        """
        if image.ndim == 3 and image.shape[2] == 1:  # Grayscale in [H, W, 1]
            image = image[:, :, 0]  # Convert to [H, W]

        smoothed = cv2.GaussianBlur(image, (0, 0), sigma)

        if smoothed.ndim == 2:  # Convert back to [H, W, 1] if necessary
            smoothed = smoothed[:, :, np.newaxis]

        return smoothed

    @staticmethod
    def compute_hessian(image, sigma):
        """
        Computes the Hessian matrix components.

        Args:
            image (numpy.ndarray): Input image (H, W) or (H, W, 1).
            sigma (float): Scale for Gaussian derivative.

        Returns:
            tuple: Hessian components (Ixx, Iyy, Ixy) with the same shape as input.
        """
        if image.ndim == 3 and image.shape[2] == 1:  # Grayscale in [H, W, 1]
            image = image[:, :, 0]  # Convert to [H, W]

        Ixx = gaussian_filter(image, sigma=sigma, order=(2, 0))
        Iyy = gaussian_filter(image, sigma=sigma, order=(0, 2))
        Ixy = gaussian_filter(image, sigma=sigma, order=(1, 1))

        if Ixx.ndim == 2:  # Convert back to [H, W, 1] if necessary
            Ixx = Ixx[:, :, np.newaxis]
            Iyy = Iyy[:, :, np.newaxis]
            Ixy = Ixy[:, :, np.newaxis]

        return Ixx, Iyy, Ixy

    @staticmethod
    def eigenvalues_of_hessian(Ixx, Iyy, Ixy):
        """
        Computes eigenvalues of the Hessian matrix.

        Args:
            Ixx, Iyy, Ixy (numpy.ndarray): Hessian components (H, W, 1).

        Returns:
            tuple: Eigenvalues (lambda1, lambda2) with the same shape as input.
        """
        trace = Ixx + Iyy
        determinant = Ixx * Iyy - Ixy**2
        discriminant = np.sqrt(np.maximum((trace / 2)**2 - determinant, 0))

        lambda1 = trace / 2 + discriminant
        lambda2 = trace / 2 - discriminant

        return lambda1, lambda2

    def vesselness_filter(self, image):
        """
        Applies the vesselness filter over multiple scales.

        Args:
            image (numpy.ndarray): Input image (H, W, 1).

        Returns:
            numpy.ndarray: Vesselness map (H, W, 1).
        """

        vesselness = np.zeros_like(image, dtype=np.float32)

        for sigma in self.scales:
            smoothed = self.apply_gaussian_smoothing(image, sigma)
            Ixx, Iyy, Ixy = self.compute_hessian(smoothed, sigma)
            lambda1, lambda2 = self.eigenvalues_of_hessian(Ixx, Iyy, Ixy)

            lambda1, lambda2 = np.maximum(lambda1, lambda2), np.minimum(lambda1, lambda2)

            Rb = np.abs(lambda1) / np.sqrt(np.abs(lambda1 * lambda2) + 1e-6)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            v = np.exp(-Rb**2 / (2 * self.beta**2)) * (1 - np.exp(-S**2 / (2 * self.c**2)))
            vesselness = np.maximum(vesselness, v)

        return vesselness, Rb, S

    def vessel_region_info(self, vesselness):
        """
        Computes the vessel region information function.

        Args:
            vesselness (numpy.ndarray): Vesselness map (H, W, 1).

        Returns:
            numpy.ndarray: Vessel region information (H, W, 1).
        """
        P = np.zeros_like(vesselness, dtype=np.float32)

        P[vesselness > self.a] = -1
        P[vesselness < self.b] = 1 - vesselness[vesselness < self.b]

        mask_middle = (vesselness >= self.b) & (vesselness <= self.a)
        P[mask_middle] = -vesselness[mask_middle]

        return P

    def __call__(self, sample):
        """
        Applies the vesselness filter and computes the vessel region information function.

        Args:
            sample (dict): Dictionary containing 'transformed_image' (H, W, 1).

        Returns:
            dict: Updated sample with 'P' added.
        """
        image = sample['transformed_image']

        # Apply the vesselness filter
        vesselness_map, Rb, S = self.vesselness_filter(image)
        sample['Rb'] = Rb
        sample['S'] = S
        sample['vesselness_map'] = vesselness_map
        P = self.vessel_region_info(vesselness_map)

        sample['P'] = P
        
        return sample


import heapq

class MPP_BT_Transform:
    def __init__(self, eta=1e-6, alpha=0.7, l_bk=15, ns_min=0.05, l_e=1500, l_ave=1000):
        """
        MPP-BT Algorithm Implementation
        :param eta: Small positive number to avoid zero division.
        :param alpha: Quantile threshold for centerline segmentation.
        :param l_bk: Backtracking step size.
        :param ns_min: Minimum normalized speed for stopping criterion.
        :param l_e: Number of consecutive points to stop propagation.
        :param l_ave: Number of points to average for speed normalization.
        """
        self.eta = eta
        self.alpha = alpha
        self.l_bk = l_bk
        self.ns_min = ns_min
        self.l_e = l_e
        self.l_ave = l_ave
        self.kappa = 1.0
        self.gamma = 100

    def compute_convexity(self, image, dx, dy):
        """
        Compute the convexity metric c(p, theta) for the given direction (dx, dy).
        :param image: 2D grayscale image.
        :param dx: Direction x-component.
        :param dy: Direction y-component.
        :return: Convexity metric.
        """
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        curvature = grad_x * dx + grad_y * dy
        return curvature

    def compute_symmetry(self, image, dx, dy):
        """
        Compute the symmetry metric s(p, theta) for the given direction (dx, dy).
        :param image: 2D grayscale image.
        :param dx: Direction x-component.
        :param dy: Direction y-component.
        :return: Symmetry metric.
        """
        shifted_image = np.roll(image, shift=(dx, dy), axis=(0, 1))
        symmetry = 1.0 / (1 + np.abs(image - shifted_image))
        return symmetry

    def potential_function(self, image):
        """
        Compute the potential function P_c(p) using medialness measure.
        :param image: 2D grayscale image.
        :return: Potential map.
        """
        h, w = image.shape
        potential_map = np.ones((h, w), dtype=np.float32)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals

        for dx, dy in directions:
            c = self.compute_convexity(image, dx, dy)
            s = self.compute_symmetry(image, dx, dy)
            mc = ((self.kappa + c)**2 * s**2) / (self.gamma * s**2 + 1)
            potential_map *= mc  # Multiply M_c(p, theta) for each direction

        # Invert to get the potential function
        potential_map = 1 / (self.eta + potential_map)
        return potential_map

    def dijkstra(self, potential_map, start_point):
        """
        Perform Dijkstra's algorithm for minimal path propagation.
        Optimized with NumPy arrays for speed.
        :param potential_map: 2D array with potential values.
        :param start_point: Tuple (x, y) representing the starting point.
        :return: Distance map and backtracking map as NumPy arrays.
        """
        h, w = potential_map.shape
        distance = np.full((h, w), np.inf, dtype=np.float32)
        distance[start_point] = 0
        visited = np.zeros((h, w), dtype=bool)
        backtrack_map = -np.ones((h, w, 2), dtype=np.int32)  # Track previous points

        priority_queue = [(0, start_point)]  # (distance, (x, y))

        while priority_queue:
            current_distance, (x, y) = heapq.heappop(priority_queue)

            if visited[x, y]:
                continue
            visited[x, y] = True

            # Check 8-connected neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    new_distance = current_distance + potential_map[nx, ny]
                    if new_distance < distance[nx, ny]:
                        distance[nx, ny] = new_distance
                        heapq.heappush(priority_queue, (new_distance, (nx, ny)))
                        backtrack_map[nx, ny] = [x, y]

        return distance, backtrack_map

    def backtrack(self, potential_map, backtrack_map):
        """
        Perform backtracking to construct the centerline feature map.
        Optimized for speed with NumPy.
        :param potential_map: 2D array with potential values.
        :param backtrack_map: 3D NumPy array with previous point indices.
        :return: Centerline feature map as a 2D NumPy array.
        """
        h, w = potential_map.shape
        I_BK = np.zeros((h, w), dtype=np.float32)

        # Traverse backtracking map
        end_points = np.argwhere(backtrack_map[:, :, 0] != -1)  # All reachable points

        # Prepare an array to mark visited points
        visited = np.zeros_like(I_BK, dtype=bool)

        # Vectorized backtracking
        for ex, ey in end_points:
            stack = [(ex, ey)]  # Use a stack for traversal
            while stack:
                cx, cy = stack.pop()
                if visited[cx, cy]:
                    continue

                visited[cx, cy] = True  # Mark as visited
                I_BK[cx, cy] += 1 / (self.eta + potential_map[cx, cy])  # Update centerline map

                # Get previous point from backtrack map
                prev_x, prev_y = backtrack_map[cx, cy]
                if prev_x != -1 and prev_y != -1:  # Check if the point is valid
                    stack.append((prev_x, prev_y))

        return I_BK

    def extract_centerline(self, potential_map):
        """
        Extract centerline using MPP-BT algorithm.
        Optimized with NumPy for speed.
        :param potential_map: 2D array with potential values.
        :return: Binary centerline map (0 for background, 1 for centerline).
        """
        # Get starting point (maximum potential value)
        start_point = np.unravel_index(np.argmax(potential_map), potential_map.shape)

        # print(f"Start point: {start_point}")

        # Perform Dijkstra's algorithm
        _, backtrack_map = self.dijkstra(potential_map, start_point)

        # print("Dijkstra's algorithm completed.")

        # Construct centerline feature map
        I_BK = self.backtrack(potential_map, backtrack_map)

        # print("Backtracking completed.")

        # Thresholding
        non_zero_values = I_BK[I_BK > 0]
        threshold = np.quantile(non_zero_values, self.alpha)
        centerline = (I_BK >= threshold).astype(np.float32)

        return centerline

    def __call__(self, sample):
        """
        Apply the MPP-BT transform to a sample.

        :param sample: Dictionary containing 'transformed_image' as a NumPy array.
        :return: Updated sample with 'centerline' as a NumPy array.
        """
        # Extract the transformed image
        transformed_image = sample['transformed_image']
        assert transformed_image.ndim == 3 and transformed_image.shape[-1] == 1, \
            "transformed_image must have shape [H, W, 1]"

        # Compute the potential function
        potential_map = self.potential_function(transformed_image[..., 0])

        # Extract centerline
        centerline = self.extract_centerline(potential_map)

        # Add centerline to the sample
        sample['centerline'] = centerline[..., None]  # Maintain [H, W, 1] format

        return sample



class ToTensor:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = transforms.ToTensor()(sample['image']).float()
        sample['mask'] = transforms.ToTensor()(sample['mask']).float()
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image']).float()
        sample['transformed_mask'] = transforms.ToTensor()(sample['transformed_mask']).float()
        sample['sdt'] = transforms.ToTensor()(sample['sdt']).float()
        sample['dt'] = transforms.ToTensor()(sample['dt']).float()
        sample['P'] = transforms.ToTensor()(sample['P']).float()
        sample['centerline'] = transforms.ToTensor()(sample['centerline']).float()
        return sample


def get_transform(args, mode='training'):
    if mode == 'training':
        transform = transforms.Compose([
            StandardTransform(),
            Keep(),
            ComputeSignedDistance(max_distance=30), # ('sdt', 'dt')
            MPP_BT_Transform(),
            VesselnessFilter(scales=[2, 4, 8, 16, 32, 64, 1], beta=5.0, c=0.005, a=0.5, b=0.2),
            ToTensor()
        ])
    else:
        transform = transforms.Compose([
            PadToSquare(),
            Keep(),
            ComputeSignedDistance(max_distance=30), # ('sdt', 'dt')
            MPP_BT_Transform(),
            VesselnessFilter(scales=[2, 4, 8, 16, 32, 64, 1], beta=5.0, c=0.005, a=0.5, b=0.2),
            ToTensor(),
        ])
    
    return transform


class DRIVEDataset(Dataset):
    def __init__(self, mode, path, opt ,is_val=False, split=None, transform=None):
        self.path = path
        self.transform = transform
        self.data_path = os.path.join(path, f"{mode}_{opt}")
        self.data_file = os.listdir(self.data_path)
        self.image_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.image_file = self.image_file[:int(split*len(self.image_file))]
            else:
                self.image_file = self.image_file[int(split*len(self.image_file)):]

    def __len__(self):
        return len(self.image_file)  # データセットのサイズを返す

    def __getitem__(self, idx):
        image_file = self.image_file[idx]
        with open(file=os.path.join(self.data_path, image_file), mode='rb') as file:
            image = pickle.load(file)
            image = np.transpose(image, (1, 2, 0))
        gt_file = "gt" + image_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            mask = pickle.load(file)
            mask = np.transpose(mask, (1, 2, 0))

        sample = {'image': image, 'mask': mask, 'meta': {'img_path': image_file, 'mask_path': gt_file}}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list