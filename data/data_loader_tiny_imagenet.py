import numpy as np
import torch
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from option import args
from PIL import Image
import os


# refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

class Dataset_npy_batch(data.Dataset):
    def __init__(self, npy_dir, transform):
        self.data = np.load(os.path.join(npy_dir, "batch.npy"))
        self.transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][0])
        img = self.transform(img)
        label = self.data[index][1]
        label = torch.LongTensor([label])[0]
        return img.squeeze(0), label

    def __len__(self):
        return len(self.data)

# def get_data_loader(data_path, batch_size):
#     train_set = datasets.ImageFolder(
#         data_path + "/train",
#         transform=transform_train,
#     )
#     test_set = datasets.ImageFolder(
#         data_path + "/val",
#         transform=transform_test,
#     )
#     return train_set, test_set

def get_data_loader(data_path, batch_size):
    # train_set = datasets.ImageFolder(
    #     data_path + "/train",
    #     transform=transform_train,
    # )
    train_set = Dataset_npy_batch(
        data_path + "/train",
        transform=transform_train,
    )
    test_set = Dataset_npy_batch(
        data_path + "/val",
        transform=transform_test,
    )
    return train_set, test_set
