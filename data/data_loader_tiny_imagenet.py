import numpy as np
import torch
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from option import args
from PIL import Image
import os
from utils import Dataset_npy_batch


# refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8), # refer to the cifar case
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def get_data_loader(data_path, batch_size):
    train_set = Dataset_npy_batch(
        data_path + "/train",
        transform=transform_train,
    )
    test_set = Dataset_npy_batch(
        data_path + "/val",
        transform=transform_test,
    )
    return train_set, test_set
