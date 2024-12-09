import os
import numpy as np
import cv2
import PIL
import json
import multiprocessing.dummy as multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader

from dataloaders.helpers import *
import dataloaders.custom_transforms as tr
from mypath import Path


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def process_info(fname, skip_multicomp=False):
    with open(fname, 'r') as f:
        ann = json.load(f)

    ret = {"objects": []}
    idx = 0
    for obj in ann:
        ret["im_size"] = (obj['img_height'], obj['img_width'])
        ret['im_path'] = obj['img_path']
        ret["label"] = "car"
        ret["idx"] = "0"
        
        if obj['label'] not in [
            "car",
            # "truck",
            # "train",
            # "bus",
            # "motorcycle",
            # "bicycle",
            # "rider",
            # "person"
        ]:
            continue

        components = obj['components']
        candidates = [c for c in components if len(c['poly']) >= 3]
        candidates = [c for c in candidates if c['area'] >= 100]

        # instance = dict()
        # polygon
        ret["objects"].append([np.array(comp['poly']) for comp in candidates])
        # instance['im_size'] = (obj['img_height'], obj['img_width'])
        # instance['im_path'] = obj['img_path']
        # instance['label'] = obj['label']
        # instance['idx'] = str(idx)
        idx += 1

        # if skip_multicomp and len(candidates) > 1:
        #     continue
        #if candidates:
        #     ret.append(instance)

    return ret


class CityScapesCustom(Dataset):
    """CityScapes dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True, split='train',
                 db_root_dir=Path.db_root_dir('cityscapes-processed'),
                 transform=None,
                 retname=True):

        self.train = train
        self.split = split
        self.db_root_dir = db_root_dir
        self.retname = retname
        self.transform = transform
        # 1要素がjson内の1オブジェクト
        self.ann_list = self.get_ann_list()

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        ann = self.ann_list[idx]
        # img = np.array(cv2.imread(ann['im_path']))
        img = np.array(PIL.Image.open(ann['im_path']).convert('RGB')).astype(np.float32)

        gt = np.zeros(ann['im_size'])
        for polygon in ann["objects"]:
            gt = cv2.fillPoly(gt, polygon, 1)

        sample = {'image': img, 'gt': gt}

        if self.retname:
            sample['meta'] = {'image': ann['im_path'].split('/')[-1][:-4],
                              'object': ann['idx'],
                              'category': ann['label'],
                              'im_size': ann['im_size']}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_ann_list(self):
        ann_list_path = os.path.join(self.db_root_dir, self.split + '_processed_ann_list.npy')
        # if os.path.exists(ann_list_path):
        #     return np.load(ann_list_path).tolist()
        # else:
        #     # preprocessing
        #     print("Preprocessing of CityScapes Dataset. This would be done only once. ")
        # data_dir = delse/dataset/cityscapes/train
        data_dir = os.path.join(self.db_root_dir, self.split)
        # 再帰的(recursive_glob)だからディレクトリ内まで潜ってjsonファイルパスのリストを作成
        ann_path_list = recursive_glob(data_dir, suffix='.json')

        # 並列化
        pool = multiprocessing.Pool(4)
        # process_infoにann_path_listの要素(json)を並列化して渡していく
        ann_list = pool.map(process_info, ann_path_list)
        # ann_list = [obj for ann in ann_list for obj in ann]
        # np.save(ann_list_path, ann_list)
        return ann_list


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import torchvision

    transforms = torchvision.transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])

    dataset = CityScapesCustom(train=True, split='train', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):
        plt.figure()
        img = tens2image(data['image']) / 255
        J = img[:, :, 0]
        J[tens2image(data['gt']) > 0.5] = 1
        plt.imshow(img)

        if i == 5:
            break