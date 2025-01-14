import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
import cv2
from PIL import Image
import numpy as np
import random
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA

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


class Normalize:
    """正規化するクラス"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = (sample['image'] / 255.0 - self.mean) / self.std
        sample['mask'] = sample['mask'] / 255.0
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


class Skeletonize:
    """マスクをスケルトン化するクラス"""
    def __call__(self, sample):
        mask = sample['mask'].squeeze()  # マスクを取得して次元を整える
        skeleton = self.skeletonize_mask(mask)
        sample['skeleton'] = skeleton
        return sample

    @staticmethod
    def skeletonize_mask(mask):
        """
        スケルトン化を行う関数
        Args:
            mask (numpy.ndarray): 入力マスク (H, W) バイナリマスク
        Returns:
            skeleton (numpy.ndarray): スケルトン (H, W) バイナリマスク
        """
        binary_mask = (mask > 0).astype(np.float16)  # バイナリマスクに変換
        return skeletonize(binary_mask)


class CalculateSkeletonDirections:
    """スケルトン上の方向ベクトルを計算するクラス"""
    def __call__(self, sample):
        skeleton = sample.get('skeleton', None)
        mask = sample['mask'].squeeze()

        if skeleton is not None:
            skeleton_directions = self.calculate_skeleton_directions(skeleton)
            vessel_directions = self.find_nearest_skeleton_directions(mask, skeleton, skeleton_directions)

            sample['skeleton_directions'] = skeleton_directions
            sample['vessel_directions'] = vessel_directions

        return sample

    @staticmethod
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

    @staticmethod
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



class ToTensor:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        # 全てのkeyに対して変換を適用
        sample['image'] = transforms.ToTensor()(sample['image']).float()
        sample['mask'] = transforms.ToTensor()(sample['mask']).float()
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image']).float()
        sample['transformed_mask'] = transforms.ToTensor()(sample['transformed_mask']).float()
        sample['skeleton'] = transforms.ToTensor()(sample['skeleton']).float()
        sample['vessel_directions'] = transforms.ToTensor()(sample['vessel_directions']).float()
        
        return sample


def get_transform(args, mode='training'):
    if mode == 'training':
        transform = transforms.Compose([
            StandardTransform(),
            PadToSquare(),
            Keep(),
            Skeletonize(),
            CalculateSkeletonDirections(),
            ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            PadToSquare(),
            Keep(),
            Skeletonize(),
            CalculateSkeletonDirections(),
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
            # assert split > 0 and split < 1
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

        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)

        sample['meta'] = {'img_path': image_file, 'mask_path': gt_file}

        return sample
    
    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list