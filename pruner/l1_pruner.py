import torch
import torch.nn as nn
import copy
import time
import numpy as np
from utils import _weights_init
from pruner import Pruner


class L1Pruner(Pruner):
    def __init__(self, model, args, logger, runner):
        super(L1Pruner, self).__init__(model, args, logger, runner)

    def prune(self):
        if self.args.arch.startswith('resnet'):
            self._get_kept_chl_L1_resnet(self.args.prune_ratio)
            self._prune_and_build_new_model()
        elif self.args.arch.startswith('alexnet') or self.args.arch.startswith('vgg'):
            self._get_kept_chl_L1(self.args.prune_ratio) # TODO: layer-wise pr
            self._prune_and_build_new_model()
        else:
            raise NotImplementedError
                    
        if self.args.reinit:
            self.model.apply(_weights_init) # equivalent to training from scratch
            self.print("Reinit model")

        return self.model
    
