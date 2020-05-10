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
        arch = self.args.arch
        if arch.startswith('resnet'):
            if arch in ['resnet18', 'resnet34']:
                self._get_kept_wg_L1_resnet_basic(self.args.prune_ratio, self.args.wg)
            elif arch in ['resnet50', 'resnet101', 'resnet152']:
                self._get_kept_wg_L1_resnet_bottleneck(self.args.prune_ratio, self.args.wg)
            else:
                raise NotImplementedError
            self._prune_and_build_new_model()

        elif arch.startswith('alexnet') or arch.startswith('vgg'):
            self._get_kept_wg_L1(self.args.prune_ratio, self.args.wg)
            self._prune_and_build_new_model()
        else:
            raise NotImplementedError
                    
        if self.args.reinit:
            self.model.apply(_weights_init) # equivalent to training from scratch
            self.print("Reinit model")

        return self.model
    
