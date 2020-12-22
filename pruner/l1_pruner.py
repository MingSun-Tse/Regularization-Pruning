import torch
import torch.nn as nn
import copy
import time
import numpy as np
from utils import _weights_init
from .meta_pruner import MetaPruner

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, runner):
        super(Pruner, self).__init__(model, args, logger, runner)

    def prune(self):
        self._get_kept_wg_L1()
        self._prune_and_build_new_model()
                    
        if self.args.reinit:
            self.model.apply(_weights_init) # equivalent to training from scratch
            self.logprint("Reinit model")

        return self.model