from matplotlib import pyplot as plt
import torch
import torchvision
from dataloaders.cityscapes_custom import CityScapesCustom
import dataloaders.custom_transforms as tr

from dataloaders.helpers import *

if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])

    dataset = CityScapesCustom(train=False, split='val', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    for i, batch in enumerate(dataloader):
        if i > 4:
            break
        
        mask = batch["gt"][0].squeeze(0)
        img = batch['image'][0].permute(1, 2, 0).numpy() / 255
        
        plt.clf()
        plt.figure(figsize=(10, 6))
    
        plt.subplot(2,1,1)
        plt.title('image')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(2,1,2)
        plt.title('mask')
        plt.imshow(mask, 'gray')
        plt.axis('off')

        plt.savefig(f"check_mask_{i}.png", bbox_inches='tight', pad_inches=0)

        plt.show()
    