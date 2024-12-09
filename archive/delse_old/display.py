# execution example
# 
# (OLD) python display.py --img_path /home/sano/dataset/cityscapes/leftImg8bit/val/munster/munster_000172_000019_leftImg8bit.png --mask_path_initial /home/sano/documents/delse_old/exp/run_0003/results_ep399/munster_000172_000019_leftImg8bit --overlay True
# (NEW) python display.py --img_path /home/sano/dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png --mask_path_initial /home/sano/documents/delse_old/exp/run_0003/results_ep399/frankfurt_000000_000294_leftImg8bit --overlay True
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()

parser.add_argument('--img_path', type=str, default='')
parser.add_argument('--mask_path_initial', type=str, default='', help='remove \"-num.npy\" from full path.')
parser.add_argument('--overlay', type=bool, default=False)

def display_parallel(img, mask, n):
    colors = plt.cm.get_cmap('tab20', n).colors
    cmap = mcolors.ListedColormap(colors)

    plt.clf()
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2,1,1)
    plt.title('image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title('mask')
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')

    plt.savefig("check_mask.png", bbox_inches='tight', pad_inches=0)

    plt.show()


def display_overlay(img, mask, n):
    colors = plt.cm.get_cmap('tab20', n).colors
    cmap = mcolors.ListedColormap(colors)

    plt.clf()
    plt.figure()

    plt.imshow(img)
    plt.imshow(mask, cmap=cmap, alpha=0.5)
    plt.axis('off')

    plt.savefig("check_mask.png", bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    img_path = args.img_path
    mask_path_initial = args.mask_path_initial

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    count_index = 1
    for index in range(100):
        mask_path = mask_path_initial + "-" + str(index) + ".npy"
        if os.path.exists(mask_path):
            print(f"exist: {index}")
            class_mask = np.load(mask_path)
            full_mask[class_mask <= -2] = (index+1)
            count_index += 1
    if args.overlay:
        display_overlay(img, full_mask, count_index)
    else:
        display_parallel(img, full_mask, count_index)