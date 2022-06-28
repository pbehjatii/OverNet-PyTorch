import numpy as np
import os
import glob
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import glob

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def mean__std(data_loader):
    cnt = 0
    mean = torch.empty(3)
    std = torch.empty(3)
    # import pdb;
    # pdb.set_trace()

    for data, label in data_loader:
        b, c, h, w = data.size()
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        mean = (cnt * mean + sum_) / (cnt + nb_pixels)
        std = (cnt * std + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return mean, torch.sqrt(std - mean ** 2)

train_data = torchvision.datasets.ImageFolder('b', transform=transforms.Compose([transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False, num_workers=4)

mean, std = mean__std(data_loader)
print(mean, std)
