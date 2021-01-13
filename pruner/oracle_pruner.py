import torch
import torch.nn as nn
import copy
import time
import numpy as np
from utils import _weights_init
from .meta_pruner import MetaPruner
import itertools

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)
        self.test_trainset = lambda net: passer.test(passer.train_loader, net, passer.criterion, passer.args)
        
    def _get_kept_wg_oracle(self):
        # get all the possible wg combinations to prune
        combinations_layer = [] # pruned index combination of each layer
        for name, m in self.model.named_modules():
            if self.pr.get(name):
                if self.args.wg == 'filter':
                    n_wg = self.layers[name].size[0]
                elif self.args.wg == 'channel':
                    n_wg = self.layers[name].size[1]
                elif self.args.wg == 'weight':
                    n_wg = np.prod(self.layers[name].size)
                n_pruned = int(n_wg * self.pr[name])
                combinations_layer.append(list(itertools.combinations(range(n_wg), n_pruned)))
        
        # orable pruning
        pruned_index_pairs = list(itertools.product(*combinations_layer))
        self.logprint('oracle pruning: %d pairs of pruned index to ablate' % len(pruned_index_pairs))
        pruned_loss = []
        cnt = 0
        for pair in pruned_index_pairs: # for each pruned index pair, get a pruned loss
            cnt += 1
            cnt_m = 0
            model = copy.deepcopy(self.model)
            for name, m in model.named_modules():
                if self.pr.get(name):
                    pruned_index = pair[cnt_m]
                    if isinstance(m, nn.Conv2d):
                        m.weight.data[pruned_index,:,:,:] = 0
                    else:
                        m.weight.data[pruned_index,:] = 0 # FC layer
                    cnt_m += 1
            acc1, acc5, loss = self.test_trainset(model)
            pruned_loss.append(loss)
            self.logprint('[%d/%d] oracle pruning. pruned loss: %.4f' % (cnt, len(pruned_index_pairs), loss))
        
        # get the pruned index pair that leads to least pruned loss
        best_pruned_index_pair = pruned_index_pairs[np.argmin(pruned_loss)]
        self.logprint('oracle pruning. picked index pair to prune: %s, the incurred loss: %.4f' % (best_pruned_index_pair, np.min(pruned_loss)))
        cnt_m = 0
        for name, m in model.named_modules():
            if name in self.pr:
                if self.args.wg == 'filter':
                    n_wg = self.layers[name].size[0]
                elif self.args.wg == 'channel':
                    n_wg = self.layers[name].size[1]
                elif self.args.wg == 'weight':
                    n_wg = np.prod(self.layers[name].size)
                
                if self.pr[name]:
                    self.pruned_wg[name] = best_pruned_index_pair[cnt_m]
                    self.kept_wg[name] = [x for x in range(n_wg) if x not in self.pruned_wg[name]]
                    cnt_m += 1
                else:
                    self.pruned_wg[name] = []
                    self.kept_wg[name] = list(range(n_wg))
    
    def prune(self):
        self._get_kept_wg_oracle()
        self._prune_and_build_new_model()
                    
        if self.args.reinit:
            self.model.apply(_weights_init) # equivalent to training from scratch
            self.logprint("Reinit model")

        return self.model