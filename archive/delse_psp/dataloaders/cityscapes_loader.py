import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import json
import numpy as np
import cv2
import random
from scipy.ndimage import distance_transform_edt
from dataloaders.class_mapping import get_class_mapping
from myconfig import config

class_mapping = get_class_mapping(config['dataset'])

def compute_signed_distance_transform(mask, max_distance=30):
    """
    符号付き距離変換を計算する関数
    :param mask: 入力マスク (numpy array)
    :param max_distance: 距離の最大値
    :return: 符号付き距離変換 (signed distance transform)
    """
    # 距離変換を計算
    dist_in = distance_transform_edt(mask)
    dist_out = distance_transform_edt(1 - mask)
    
    # 符号付き距離変換
    signed_distance_map = dist_out - dist_in
    
    # 最大値でクリッピング
    signed_distance_map = np.clip(signed_distance_map, -max_distance, max_distance)

    return signed_distance_map

def compute_distance_transform(mask):
    """
    通常の距離変換を計算する関数
    :param mask: 入力マスク (numpy array)
    :return: 距離変換 (distance transform)
    """
    return distance_transform_edt(mask)


class HorizontalFlip:
    """元の画像とマスクに対して確率的に水平方向の反転を適用するクラス"""
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        # 確率的に元の画像とマスクを水平反転
        if np.random.rand() < self.flip_prob:
            sample['image'] = transforms.functional.hflip(sample['image'])
            sample['mask'] = np.fliplr(sample['mask']).copy()  # NumPyの関数でマスクを水平反転
        return sample

class RandomRotate:
    """画像とマスクに対してランダムな回転を適用するクラス"""
    def __init__(self, angle_range=(-10, 10)):
        self.angle_range = angle_range

    def __call__(self, sample):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        sample['image'] = transforms.functional.rotate(sample['image'], angle)
        sample['mask'] = np.array(transforms.functional.rotate(Image.fromarray(sample['mask']), angle))
        return sample

class Resize:
    """リサイズ後の画像とマスクを保存するクラス"""
    def __init__(self, size):
        self.resize = transforms.Resize(size)

    def __call__(self, sample):
        # リサイズされた画像を保存
        sample['transformed_image'] = self.resize(sample['image'])
        # マスクをPIL画像に変換してリサイズし、再度NumPy配列に変換
        resized_mask = self.resize(Image.fromarray(sample['mask']))
        sample['transformed_mask'] = np.array(resized_mask)
        return sample

class RandomResizedCrop:
    """画像とマスクにランダムなリサイズとクロップを適用するクラス"""
    def __init__(self, size, scale=(0.5, 1.0)):
        """
        :param size: リサイズ後の出力サイズ (h, w)
        :param scale: 元画像の面積に対するクロップサイズの比率範囲
        """
        self.size = size
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # ランダムクロップのパラメータを決定
        i, j, h, w = self.get_params(image, self.scale)

        # 画像とマスクに同じリサイズとクロップを適用
        transformed_image = F.resized_crop(image, i, j, h, w, self.size)
        transformed_mask = F.resized_crop(Image.fromarray(mask), i, j, h, w, self.size)

        sample['transformed_image'] = transformed_image
        sample['transformed_mask'] = np.array(transformed_mask)

        return sample

    @staticmethod
    def get_params(img, scale):
        """クロップのパラメータをランダムに決定する"""
        width, height = F.get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area

            w = int(round(np.sqrt(target_area)))
            h = int(round(np.sqrt(target_area)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # クロップが失敗した場合のフォールバック (中心クロップ)
        w, h = width, height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class ComputeSignedDistance:
    """マスクからクラスごとに符号付距離マップと距離変換を計算するクラス"""
    def __init__(self, max_distance=30, num_classes=19):
        # 符号付距離の上限を設定
        self.max_distance = max_distance
        self.num_classes = num_classes

    def __call__(self, sample):
        transformed_mask = sample['transformed_mask']
        sdt_list = []
        dt_list = []

        # 各クラスごとに符号付距離と距離変換を計算
        for class_id in range(self.num_classes):
            class_mask = (transformed_mask == class_id).astype(np.uint8)  # クラスごとのマスクを作成

            # クラスの内外の距離を計算
            dist_in = distance_transform_edt(class_mask)
            dist_out = distance_transform_edt(1 - class_mask)

            # 符号付距離を計算して上限を設定
            signed_distance_map = dist_out - dist_in
            signed_distance_map = np.clip(signed_distance_map, -self.max_distance, self.max_distance)

            # 各クラスごとの符号付距離と距離変換をリストに追加
            sdt_list.append(signed_distance_map)
            dt_list.append(dist_in)

        # 符号付距離と距離変換をクラス次元で結合して [num_classes, H, W] に変換
        signed_distance_maps = np.stack(sdt_list, axis=0)
        distance_transform_maps = np.stack(dt_list, axis=0)
        
        # サンプルに符号付距離と距離変換を追加
        sample['sdt'] = signed_distance_maps  # 符号付距離
        sample['dt'] = distance_transform_maps  # 距離変換

        return sample


class ToTensor:
    """元の画像とマスク、およびリサイズ後の画像とマスク、符号付距離と距離変換をテンソルに変換するクラス"""
    def __call__(self, sample):
        # 元の画像をテンソルに変換
        sample['image'] = transforms.ToTensor()(sample['image'])

        # オリジナルのマスクをクラスごとに0,1のバイナリマスクに変換
        h, w = sample['mask'].shape
        num_classes = len(class_mapping)
        binary_mask = np.zeros((num_classes, h, w), dtype=np.uint8)

        for class_id in range(num_classes):
            binary_mask[class_id] = (sample['mask'] == class_id).astype(np.uint8)

        # [num_classes, H, W] の形状でテンソルに変換
        sample['mask'] = torch.from_numpy(binary_mask).float()  # ピクセル値はfloatに

        # リサイズ後の画像もテンソルに変換
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image'])

        # リサイズ後のマスクも同様にクラスごとのバイナリマスクに変換
        h, w = sample['transformed_mask'].shape
        binary_transformed_mask = np.zeros((num_classes, h, w), dtype=np.uint8)

        for class_id in range(num_classes):
            binary_transformed_mask[class_id] = (sample['transformed_mask'] == class_id).astype(np.uint8)

        # [num_classes, H, W] の形状でテンソルに変換
        sample['transformed_mask'] = torch.from_numpy(binary_transformed_mask).float()

        # 符号付距離（sdt）と距離変換（dt）もテンソルに変換
        sample['sdt'] = torch.from_numpy(sample['sdt']).float()
        sample['dt'] = torch.from_numpy(sample['dt']).float()

        return sample

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.images_dir = os.path.join(root, split)
        self.mask_dir = self.images_dir.replace('leftImg8bit', 'gtFine')
        self.transform = transform

        self.images = []
        self.masks = []

        for city in os.listdir(self.images_dir):
            city_images = os.listdir(os.path.join(self.images_dir, city))
            for image in city_images:
                img_path = os.path.join(self.images_dir, city, image)
                mask_path = os.path.join(self.mask_dir, city, image.replace('_leftImg8bit.png', '_gtFine_polygons.json'))
                self.images.append(img_path)
                self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        original_image = Image.open(img_path).convert("RGB")
        with open(mask_path) as f:
            mask_json = json.load(f)
        original_mask = np.ones((mask_json['imgHeight'], mask_json['imgWidth']), dtype=np.int8) * 255
        for obj in mask_json['objects']:
            polygon = np.array(obj['polygon'], dtype=np.int32)
            label = self.get_label_id(obj['label'])
            if label != 255: # 無視クラスはスキップ
                cv2.fillPoly(original_mask, [polygon], label)

        sample = {
            'image': original_image,
            'mask': original_mask,
            'meta': {'img_path': img_path, 'mask_path': mask_path}
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_label_id(self, label):
        """ラベルからIDを取得するメソッド"""
        label_map = {
            "unlabeled": 0, "ego vehicle": 1, "rectification border": 2, "out of roi": 3, 
            "static": 4, "dynamic": 5, "ground": 6, "road": 7, "sidewalk": 8, "parking": 9,
            "rail track": 10, "building": 11, "wall": 12, "fence": 13, "pole": 17, 
            "traffic light": 19, "traffic sign": 20, "vegetation": 21, "terrain": 22, 
            "sky": 23, "person": 24, "rider": 25, "car": 26, "truck": 27, "bus": 28, 
            "train": 31, "motorcycle": 32, "bicycle": 33
        }
        original_label = label_map.get(label, 255)  # 未定義のラベルは255（無視クラス）を返す
        return class_mapping.get(original_label, 255)  # 再マッピング

def get_dataloader(split='train'):
    """transform = transforms.Compose([
        RandomRotate(angle_range=(-10, 10)),
        HorizontalFlip(flip_prob=0.5),
        Resize(size=(512, 1024)),
        ToTensor()
    ])"""
    
    if split=='train':
        transform = transforms.Compose([
            RandomRotate(angle_range=(-10, 10)),    # ランダム回転
            HorizontalFlip(flip_prob=0.5),          # 水平反転
            RandomResizedCrop(size=(256, 512)),   # ランダムクロップとリサイズ (一度廃止)
            # Resize(size=(256, 512)),                # 固定サイズにリサイズ
            ComputeSignedDistance(max_distance=30), # 符号付距離計算
            ToTensor()                              # テンソルに変換
        ])
    else:
        transform = transforms.Compose([
            Resize(size=(256, 512)),                # 固定サイズにリサイズ
            ComputeSignedDistance(max_distance=30), # 符号付距離計算
            ToTensor()                              # テンソルに変換
        ])

    if split == "train":
        dataset = CityscapesDataset(root=config["train_data"], split=split, transform=transform)
        return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    elif split == "val":
        dataset = CityscapesDataset(root=config["val_data"], split=split, transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    elif split == "test":
        dataset = CityscapesDataset(root=config["test_data"], split=split, transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])