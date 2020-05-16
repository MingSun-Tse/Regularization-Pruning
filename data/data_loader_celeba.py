import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
pjoin = os.path.join


def is_img(x):
    _, ext = os.path.splitext(x)
    return ext.lower() in ['.jpg', '.png', '.bmp', '.jpeg']


class CelebA(data.Dataset):
    '''
      Only for the most balanced attribute "Attractive".
      Deprecated. This class is not fully worked through. Be careful.
    '''

    def __init__(self, img_dir, label_file, transform):
        self.img_list = [os.path.join(img_dir, i) for i in os.listdir(
            img_dir) if i.endswith(".npy")]
        self.transform = transform
        if label_file.endswith(".npy"):
            self.label = np.load(label_file)  # label file is npy
        else:
            self.label = {}
            for line in open(label_file):  # label file is txt
                if ".jpg" not in line:
                    continue
                img_name, *attr = line.strip().split()
                # "Attractive" is at the 3rd position of all attrs
                self.label[img_name] = int(attr[2] == "1")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = img_path.split("/")[-1]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))  # for alexnet
        img = self.transform(img)
        return img.squeeze(0), self.label[img_name]

    def __len__(self):
        return len(self.img_list)


class CelebA_npy(data.Dataset):
    def __init__(self, npy_dir, label_file, transform):
        self.npy_list = [os.path.join(npy_dir, i) for i in os.listdir(
            npy_dir) if i.endswith(".npy") and i != "batch.npy"]
        self.transform = transform
        # label_file should be an npy
        self.label = torch.from_numpy(np.load(label_file)).long()

    def __getitem__(self, index):
        npy = self.npy_list[index]
        img = np.load(npy)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img.squeeze(0), self.label[int(npy.split("/")[-1].split(".")[0])]

    def __len__(self):
        return len(self.npy_list)


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


def get_data_loader(data_path, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_data_path = pjoin(data_path, "train_npy")
    train_label_path = pjoin(data_path, "CelebA_Attractive_label.npy")
    test_path = pjoin(data_path, "test_npy")
    assert(os.path.exists(train_data_path))
    assert(os.path.exists(train_label_path))
    assert(os.path.exists(test_path))
    
    train_set = CelebA_npy(
        train_data_path, train_label_path, transform=transform_train)
    test_set = Dataset_npy_batch(test_path, transform=transform_test)
    

    return train_set, test_set