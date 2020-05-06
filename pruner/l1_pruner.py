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
    
    def _pick_chl(self, w_abs, pr, mode="min"):
        C = len(w_abs.flatten())
        if mode == "rand":
            out = np.random.permutation(C)[:int(pr * C)]
        elif mode == "min":
            out = w_abs.flatten().sort()[1][:int(pr * C)]
        elif mode == "max":
            out = w_abs.flatten().sort()[1][-int(pr * C):]
        return out

    def _get_kept_chl(self, prune_ratios):
        conv_cnt = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d): # for now, we focus on conv layers
                conv_cnt += 1
                C = m.weight.size(1)
                if conv_cnt in [1]:
                    self.pruned_chl[m] = []
                else:
                    if isinstance(prune_ratios, dict):
                        pr = prune_ratios[m]
                    else:
                        pr = prune_ratios
                    w_abs = m.weight.abs().mean(dim=[0, 2, 3])
                    self.pruned_chl[m] = self._pick_chl(w_abs, pr, self.args.pick_pruned)
                self.kept_chl[m] = [i for i in range(C) if i not in self.pruned_chl[m]]

    def _get_kept_chl_resnet(self, prune_ratios):
        conv_cnt = 0
        just_passed_3x3 = False
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1
                C = m.weight.size(1)
                w_abs = m.weight.abs().mean(dim=[0, 2, 3]) 
                pr = prune_ratios
                if m.kernel_size == (3, 3):
                    self.pruned_chl[m] = self._pick_chl(w_abs, pr, self.args.pick_pruned)
                    just_passed_3x3 = True
                elif  m.kernel_size == (1, 1) and just_passed_3x3:
                    self.pruned_chl[m] = self._pick_chl(w_abs, pr, self.args.pick_pruned)
                    just_passed_3x3 = False
                else: # all the first 1x1 conv layers and non-3x3 conv layers
                    self.pruned_chl[m] = []
                self.kept_chl[m] = [i for i in range(C) if i not in self.pruned_chl[m]]

    def prune(self):
        if self.args.arch.startswith('resnet'):
            self._get_kept_chl_resnet(self.args.prune_ratio)
            self._prune_and_build_new_model_resnet()
        else:
            self._get_kept_chl(self.args.prune_ratio) # TODO: layer-wise pr
            self._prune_and_build_new_model_resnet()
                    
        if self.args.reinit:
            self.model.apply(_weights_init) # equivalent to training from scratch
            self.print("Reinit model")

        return self.model
    
