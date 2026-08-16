import torch
import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode



class SegmentationTransform:

    def __init__(self, image_size=256):

        self.image_size = image_size


    def __call__(self, image, mask):


        image = TF.resize(
            image,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )


        mask = TF.resize(
            mask,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )



        image = TF.to_tensor(image)


        image = TF.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )



        mask = np.array(mask)


        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )


        mask = mask / 255.0 #değeri 0-1 e indiriyoruz normalde 0-255 artık 0-1


        mask = mask.unsqueeze(0) # bu bize bir boyut daha ekliyor


        return image, mask




class PolypDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform


        self.image_dir = os.path.join(
            root_dir,
            "images"
        )


        self.mask_dir = os.path.join(
            root_dir,
            "masks"
        )


        self.images = sorted(
            os.listdir(self.image_dir)
        )



    def __len__(self):

        return len(self.images)



    def __getitem__(self, index):


        image_path = os.path.join(
            self.image_dir,
            self.images[index]
        )


        mask_path = os.path.join(
            self.mask_dir,
            self.images[index]
        )



        image = Image.open(
            image_path
        ).convert("RGB")


        mask = Image.open(
            mask_path
        ).convert("L")



        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask
            )


        return image, mask




def get_data_loaders(batch_size=8, image_size=256):


    segmentation_transform = SegmentationTransform(
        image_size=image_size
    )



    train_set = PolypDataset(
        root_dir="./dataset_split/train",
        transform=segmentation_transform
    )


    test_set = PolypDataset(
        root_dir="./dataset_split/test",
        transform=segmentation_transform
    )



    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )



    return train_loader, test_loader