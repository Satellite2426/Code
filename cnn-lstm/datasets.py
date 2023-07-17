import pickle
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms as transforms


class MyDataset(Dataset):

    def __init__(self, datas):
        super(MyDataset, self).__init__()
        self.datas = datas

        self.color_transforms = transforms.Compose([  # color图片处理
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.depth_transforms = transforms.Compose([  # 深度图片处理
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, idx):
        data = self.datas[idx]

        noise_data = torch.Tensor(data['noise_data']) / 100.
        point_data = torch.Tensor(data['point_data']) / 100.

        color_data = Image.fromarray(data['color_data'])
        color_data = self.color_transforms(color_data) / 255.  # 归一化

        depth_data = Image.fromarray(data['depth_data'])
        depth_data = self.depth_transforms(depth_data) / 1000.  # 归一化

        label = torch.Tensor(data['label']) / 100.

        return noise_data, point_data, color_data, depth_data, label

    def __len__(self):
        return len(self.datas)
