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
            self._get_kept_wg_L1_resnet()
        elif arch.startswith('alexnet') or arch.startswith('vgg'):
            self._get_kept_wg_L1()
        else:
            raise NotImplementedError
        self._prune_and_build_new_model()
                    
        if self.args.reinit:
            self.model.apply(_weights_init) # equivalent to training from scratch
            self.print("Reinit model")

        return self.model
    
