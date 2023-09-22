from enum import Enum

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

from .build_dataset import abs_path, get_dataset_list


class InputChannelConfig(Enum):
    MASK_ONLY = 1  # single channel
    RGB_ONLY = 3  # 3 channels
    RGB_MASK = 4  # 4 channels


class FluvialDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None, target_transform=None,
                 channel_config=InputChannelConfig.RGB_ONLY):
        """
        Custom fluvial dataset class
        :param dataset_path: relative path of target dataset csv file in src/dataset/
        :param transform: original image transform
        :param target_transform: mask image transform
        """
        super(FluvialDataset, self).__init__()
        dataset_file = abs_path(dataset_path)
        dataset_list = get_dataset_list(dataset_file)

        self.channel_config = channel_config

        if channel_config != InputChannelConfig.MASK_ONLY:
            self.img_files = [pair[0] for pair in dataset_list]
        else:
            self.img_files = None

        if channel_config != InputChannelConfig.RGB_ONLY:
            self.mask_files = [pair[1] for pair in dataset_list]
        else:
            self.mask_files = None
        # print(self.img_files)
        # print(self.mask_files)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image, mask = None, None
        if self.img_files:
            # print(f'image path is {self.img_files[idx]}')
            image = read_image(self.img_files[idx])
            if self.transform:
                image = self.transform(image)
                image = image / 255
                # print(f"image shape {image.shape}")

        if self.mask_files:
            mask = read_image(self.mask_files[idx])
            if self.target_transform:
                mask = self.target_transform(mask)
                # convert 3-channel gray mask to 1-channel mask
                if mask.shape[0] == 4:
                    mask = mask[:3, ...]
                if mask.shape[0] == 3:
                    mask = T.Grayscale(num_output_channels=1)(mask)  # [1 x H x W]
                # mask = mask.squeeze()  # remove redundant dimension
                mask = mask / 255
                # print(f"mask shape {mask.shape}")

        if self.channel_config == InputChannelConfig.RGB_ONLY:
            assert image is not None
            return image
        elif self.channel_config == InputChannelConfig.MASK_ONLY:
            assert mask is not None
            return mask
        elif self.channel_config == InputChannelConfig.RGB_MASK:
            assert image is not None and mask is not None and \
                   image.shape[1:] == mask.shape[1:], f'Image shape {image.shape[1:]} does not mask shape {mask.shape[1:]}'
            return torch.cat((image, mask), 0)
        else:
            print(f'Unrecognized input channel type {self.channel_config}!')
            return None

