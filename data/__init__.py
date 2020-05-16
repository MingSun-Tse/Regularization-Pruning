

from importlib import import_module
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader


# ref: https://github.com/MingSun-Tse/pytorch-AdaIN/blob/02ae320345232983c754ea233613aedc21e4d348/sampler.py
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            # np.random.seed() # this will generate new rand seed, contradicting with the rand seed assigned in trainer.py
            order = np.random.permutation(n) # shuffle
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class Data(object):
    def __init__(self, args):
        self.args = args
        loader = import_module("data.data_loader_%s" % args.dataset)
        path = os.path.join(args.data_path, args.dataset)
        train_set, test_set \
            = loader.get_data_loader(path, args.batch_size)
        
        self.train_loader = DataLoader(train_set,
                                       batch_size=args.batch_size,
                                       # sampler=InfiniteSamplerWrapper(train_set),
                                       num_workers=4,
                                       shuffle=True,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                      batch_size=256,
                                      num_workers=4,
                                      shuffle=False,
                                      pin_memory=True)
        
        '''
            TODO: if num_workers is not 0, this will cause memory error.
            ref to: https://github.com/pytorch/pytorch/issues/1355
            Still do not know why.
        '''

        if self.args.method != "ccl":
            self.train_loader_iter = iter(self.train_loader)

        # import torch
        # a = torch.randn(256, 100).cuda()
        # print(a[:1])

    def next_batch_data(self):
        if self.args.method.startswith("ccl"):
            x = torch.randn([self.args.batch_size_noise, self.args.dim_z]).cuda()
            y = torch.zeros(self.args.batch_size_noise).cuda() # placeholder, useless
            return x, y
        
        if self.args.method == "defense":
            x1 = torch.randn([self.args.batch_size_noise, self.args.dim_z])
            y1 = torch.zeros(self.args.batch_size_noise) # placeholder, useless
            x2, y2 = self.train_loader_iter.next()
            return [x1.cuda(), x2.cuda()], [y1.cuda(), y2.cuda()]
       
        else:
            x, y = self.train_loader_iter.next()
            return x.cuda(), y.cuda()
