from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_data_loader(data_path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)), # ref to: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]) # these mean and var are from official PyTorch ImageNet example
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    train_set = CIFAR10(data_path,
                        train=True,
                        download=True,
                        transform=transform_train)
    test_set = CIFAR10(data_path,
                       train=False,
                       download=True,
                       transform=transform_test)

    return train_set, test_set
