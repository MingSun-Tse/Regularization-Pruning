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
        self._register_layer_kernel_size()

        self.pruned_chl = {}
        self.kept_chl = {}
    
    def _check_data_parallel(self):
        self.data_parallel = False
        if hasattr(self.model, "features"): # alexnet and vgg
            if isinstance(self.model.features, nn.DataParallel):
                self.n_seqs = len(self.model.features.module)
                self.model.features = self.model.features.module
                self.data_parallel = True
            else:
                self.n_seqs = len(self.model.features)
        else:
            if isinstance(self.model, nn.DataParallel):
                self.data_parallel = True
                self.model = self.model.module
    
    def _set_data_parallel(self):
        if hasattr(self.model, "features"):
            self.model.features = nn.DataParallel(self.model.features)
        else:
            self.model = nn.DataParallel(self.model).cuda()
    
    def _register_layer_kernel_size(self):
        self.layer_kernel_size = OrderedDict()
        ix = -1
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

    # build a new model excluding the pruned channels
    def _prune_and_build_new_model(self):
        new_model = copy.deepcopy(self.model)
        for ix in range(self.n_seqs):
            m = self.model.features[ix]
            if isinstance(m, nn.Conv2d):
                kept_chl = self.kept_chl[m]
                
                # to see if rows need pruning
                next_conv = self._next_conv(self.model, m)
                if not next_conv:
                    kept_row_index = range(m.weight.size(0))
                else:
                    kept_row_index = self.kept_chl[next_conv]
                
                # slice out kept weights and biases for conv
                kept_weights = m.weight.data[kept_row_index][:, kept_chl, :, :]
                new_conv = nn.Conv2d(kept_weights.size(1), kept_weights.size(0), m.kernel_size,
                                  m.stride, m.padding, m.dilation, m.groups, len(m.bias)).cuda()
                new_conv.weight.data.copy_(kept_weights) # load weights into the new module
                if len(m.bias):
                    kept_bias = m.bias.data[kept_row_index]
                    new_conv.bias.data.copy_(kept_bias)
                new_model.features[ix] = new_conv
                
                # get the corresponding bn layer, where the input_channel should be adjusted as well
                k = 1
                while ix + k < self.n_seqs and (not isinstance(self.model.features[ix + k], nn.BatchNorm2d)):
                    if isinstance(self.model.features[ix + k], nn.Conv2d):
                        break
                    else:
                        k += 1
                if ix + k < self.n_seqs:
                    if isinstance(self.model.features[ix + k], nn.BatchNorm2d):
                        bn = self.model.features[ix + k]
                        new_bn = nn.BatchNorm2d(len(kept_row_index), eps=bn.eps, momentum=bn.momentum, 
                                affine=bn.affine, track_running_stats=bn.track_running_stats).cuda()
                        
                        # copy bn weight and bias
                        if self.args.copy_bn_w:
                            weight = bn.weight.data[kept_row_index]
                            new_bn.weight.data.copy_(weight)
                        if self.args.copy_bn_b:
                            bias = bn.bias.data[kept_row_index]
                            new_bn.bias.data.copy_(bias)
                        
                        # copy bn running stats
                        new_bn.running_mean.data.copy_(bn.running_mean[kept_row_index])
                        new_bn.running_var.data.copy_(bn.running_var[kept_row_index])
                        new_bn.num_batches_tracked.data.copy_(bn.num_batches_tracked)
                        new_model.features[ix + k] = new_bn
        
        self.model = new_model
        t1 = time.time()
        acc1 = self.test(self.model)
        self.print("==> After  build_new_model, test acc = %.4f (time = %.2fs)" % (acc1, time.time()-t1))

    def _next_conv(self, model, m_name, mm):
        layer_names = list(self.layer_kernel_size.keys())
        layer_ix, ks = self.layer_kernel_size[m_name]
        layer_before = layer_names[layer_ix - 1]
        ks_before = self.layer_kernel_size[layer_before][1]
        if ks == (1, 1) and ks_before == (3, 3):
            return None # the last 1x1 conv in a bottleneck layer #TODO: check if this works with other resnets

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

    def _prune_and_build_new_model_resnet(self):
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

        t1 = time.time()
        acc1 = self.test(self.model)
        self.print("==> After  build_new_model, test acc = %.4f (time = %.2fs)" % (acc1, time.time()-t1))