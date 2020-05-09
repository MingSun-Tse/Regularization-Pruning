import torch
import torch.nn as nn
import copy
import time
import numpy as np
from collections import OrderedDict

class Pruner:
    def __init__(self, model, args, logger, runner):
        self.model = model
        self.args = args
        self.logger = logger
        self.print = logger.log_printer
        self.test = lambda net: runner.test(runner.test_loader, net, runner.criterion, runner.args)
        self.train_loader = runner.train_loader
        self.criterion = runner.criterion
        self._register_layer_kernel_size()

        self.pruned_chl = {}
        self.kept_chl = {}

    def _pick_chl(self, w_abs, pr, mode="min"):
        C = len(w_abs.flatten())
        if mode == "rand":
            out = np.random.permutation(C)[:int(pr * C)]
        elif mode == "min":
            out = w_abs.flatten().sort()[1][:int(pr * C)]
        elif mode == "max":
            out = w_abs.flatten().sort()[1][-int(pr * C):]
        return out

    def _register_layer_kernel_size(self):
        self.layer_kernel_size = OrderedDict()
        ix = -1 # layer index
        max_len = 0
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                ix += 1
                ks = m.kernel_size
                self.layer_kernel_size[name] = [ix, ks]
                max_len = max(max_len, len(name))

        print("Registering conv layer index and kernel size:")
        format_str = "[%3d] %{}s -- kernel_size: %s".format(max_len)
        for name, (ix, ks) in self.layer_kernel_size.items():
            print(format_str % (ix, name, ks))

    def _next_conv(self, model, m_name, mm):
        layer_names = list(self.layer_kernel_size.keys())
        layer_ix, ks = self.layer_kernel_size[m_name]
        layer_prev = layer_names[layer_ix - 1]
        layer_next = layer_names[layer_ix + 1] if layer_ix + 1 < len(layer_names) else ""
        ks_prev = self.layer_kernel_size[layer_prev][1]
        if ks == (1, 1) and ks_prev == (3, 3):
            return None # the last 1x1 conv in a bottleneck layer #TODO: check if this works with other resnets
        elif 'downsample' in layer_next:
            return None

        ix_conv = 0
        ix_mm = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                ix_conv += 1
                if m == mm:
                    ix_mm = ix_conv
                if ix_mm != -1 and ix_conv == ix_mm + 1:
                    return m
        return None
    
    def _next_bn(self, model, mm):
        just_passed_mm = False
        for m in model.modules():
            if m == mm:
                just_passed_mm = True
            if just_passed_mm and isinstance(m, nn.BatchNorm2d):
                return m
        return None
   
    def _replace_module(self, model, m_name, new_m):
        '''
            Replace the module <m_name> in <model> with <new_m>
            E.g., 'module.layer1.0.conv1'
            ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
        '''
        obj = model
        segs = m_name.split(".")
        for ix in range(len(segs)):
            s = segs[ix]
            if ix == len(segs) - 1: # the last one
                if s.isdigit():
                    obj.__setitem__(int(s), new_m)
                else:
                    obj.__setattr__(s, new_m)
                return
            if s.isdigit():
                obj = obj.__getitem__(int(s))
            else:
                obj = obj.__getattr__(s)

    def _get_kept_chl_L1(self, prune_ratios):
        '''
            Not considered dependence among layers. TODO: consider dependence.
        '''
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

    def _get_kept_chl_L1_resnet_bottleneck(self, prune_ratios):
        '''
            For pruning resnet50, 101, 152, which adopt the bottleneck block.
        '''
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

    def _get_kept_chl_L1_resnet_basic(self, prune_ratios):
        '''
            For pruning resnet18, 34, which adopt the basic block.
        '''
        conv_cnt = 0
        is_second_3x3 = False
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1
                C = m.weight.size(1) # num of channel
                w_abs = m.weight.abs().mean(dim=[0, 2, 3]) 
                pr = prune_ratios
                if m.kernel_size == (3, 3):
                    if is_second_3x3:
                        self.pruned_chl[m] = self._pick_chl(w_abs, pr, self.args.pick_pruned)
                        is_second_3x3 = False
                    else:
                        self.pruned_chl[m] = []
                        is_second_3x3 = True
                else: # 1st conv layer and 1x1 layers
                    self.pruned_chl[m] = []
                self.kept_chl[m] = [i for i in range(C) if i not in self.pruned_chl[m]]
                
    def _prune_and_build_new_model(self):
        new_model = copy.deepcopy(self.model)
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                kept_chl = self.kept_chl[m]

                # to see if rows need pruning
                next_conv = self._next_conv(self.model, name, m)
                if not next_conv:
                    kept_row_index = range(m.weight.size(0))
                else:
                    kept_row_index = self.kept_chl[next_conv]
                
                # copy conv weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                kept_weights = m.weight.data[kept_row_index][:, kept_chl, :, :]
                new_conv = nn.Conv2d(kept_weights.size(1), kept_weights.size(0), m.kernel_size,
                                  m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                new_conv.weight.data.copy_(kept_weights) # load weights into the new module
                if bias:
                    kept_bias = m.bias.data[kept_row_index]
                    new_conv.bias.data.copy_(kept_bias)
                
                # load the new conv
                self._replace_module(new_model, name, new_conv)

                # get the corresponding bn (if any) for later use
                next_bn = self._next_bn(self.model, m)

            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                new_bn = nn.BatchNorm2d(len(kept_row_index), eps=m.eps, momentum=m.momentum, 
                        affine=m.affine, track_running_stats=m.track_running_stats).cuda()
                
                # copy bn weight and bias
                if self.args.copy_bn_w:
                    weight = m.weight.data[kept_row_index]
                    new_bn.weight.data.copy_(weight)
                if self.args.copy_bn_b:
                    bias = m.bias.data[kept_row_index]
                    new_bn.bias.data.copy_(bias)
                
                # copy bn running stats
                new_bn.running_mean.data.copy_(m.running_mean[kept_row_index])
                new_bn.running_var.data.copy_(m.running_var[kept_row_index])
                new_bn.num_batches_tracked.data.copy_(m.num_batches_tracked)
                
                # load the new bn
                self._replace_module(new_model, name, new_bn)
        
        self.model = new_model
        print(self.model)

        t1 = time.time()
        acc1, acc5 = self.test(self.model)
        self.print("==> After building the new model, acc1 = %.4f, acc5 = %.4f (time = %.2fs)" % (acc1, acc5, time.time()-t1))