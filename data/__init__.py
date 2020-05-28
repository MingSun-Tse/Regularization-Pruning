

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
        self.test_loader = DataLoader(test_set,
                                      batch_size=256,
                                      num_workers=4,
                                      shuffle=False,
                                      pin_memory=True)