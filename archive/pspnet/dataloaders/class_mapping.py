# CityScapes
cityscapes_class_names = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
cityscapes_class_mapping = {
    7: 0,   # "road"
    8: 1,   # "sidewalk"
    11: 2,  # "building"
    12: 3,  # "wall"
    13: 4,  # "fence"
    17: 5,  # "pole"
    19: 6,  # "traffic light"
    20: 7,  # "traffic sign"
    21: 8,  # "vegetation"
    22: 9,  # "terrain"
    23: 10, # "sky"
    24: 11, # "person"
    25: 12, # "rider"
    26: 13, # "car"
    27: 14, # "truck"
    28: 15, # "bus"
    31: 16, # "train"
    32: 17, # "motorcycle"
    33: 18  # "bicycle"
}

# Drive
drive_class_names = [
    "vessel"
]

drive_class_mapping = {
    1: 1 # vessel
}

def get_class_names(dataset='cityscapes'):
    """データセット名に基づきクラス名リストを返す"""
    if dataset == 'cityscapes':
        return cityscapes_class_names
    elif dataset == 'drive':
        return drive_class_names
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_class_mapping(dataset='cityscapes'):
    """データセット名に基づきクラスマッピングを返す"""
    if dataset == 'cityscapes':
        return cityscapes_class_mapping
    elif dataset == 'drive':
        return drive_class_mapping
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_num_classes(dataset='cityscapes'):
    """データセット名に基づきクラス数を返す"""
    if dataset == 'cityscapes':
        return len(cityscapes_class_names)
    elif dataset == 'drive':
        return 1
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
