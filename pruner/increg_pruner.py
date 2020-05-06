import torch.nn as nn
import copy
import time
import numpy as np
from pruner import Pruner

class IncRegPruner(Pruner):
    def __init__(self, model, args, logger, runner):
        super(IncRegPruner, self).__init__(model, args, logger, runner)

        # IncReg related variables
        self.reg = {}
        self.delta_reg = {}
        self.hist_mag_ratio = {}
        self.n_update_reg = {}
        self.iter_update_reg_finished = {}
        self.iter_pick_pruned_finished = {}
        self.original_w_mag = {}
        self.ranking = {}
        self.pruned_chl = {}
        self.pruned_chl_L1 = {}
        self.kept_chl = {}
        self.all_layer_finish_picking = False

    def _update_reg(self):
        pass

    def _get_delta_reg(self):
        pass

    def prune(self):
        pass