

from importlib import import_module
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader


class Data(object):
    def __init__(self, args):
        self.args = args
        loader = import_module("data.data_loader_%s" % args.dataset)
        path = os.path.join(args.data_path, args.dataset)
        train_set, test_set = loader.get_data_loader(path, args.batch_size)
        
        self.train_loader = DataLoader(train_set,
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       shuffle=True,
                                       pin_memory=True)
        self.train_loader_prune = DataLoader(train_set,
                                       batch_size=args.batch_size_prune,
                                       num_workers=4,
                                       shuffle=True,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                      batch_size=256,
                                      num_workers=4,
                                      shuffle=False,
                                      pin_memory=True)

num_classes_dict = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet_subset_200': 200,
    'tiny_imagenet': 200,
}

img_size_dict = {
    'mnist': 32,
    'cifar10': 32,
    'cifar100': 32,
    'imagenet': 224,
    'tiny_imagenet': 64,
}