import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import glob
import os
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0),  A.size(1) * B.size(1))
    
    
def np_to_torch(x):
    '''
        np array to pytorch float tensor
    '''
    x = np.array(x)
    x= torch.from_numpy(x).float()
    return x

def kd_loss(y, teacher_scores, temp=1):
    p = F.log_softmax(y / temp, dim=1)
    q = F.softmax(teacher_scores / temp, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) / y.shape[0]
    return l_kl

def test(net, test_loader):
    n_example_test = 0
    total_correct = 0
    avg_loss = 0
    net.eval()
    with torch.no_grad():
        pred_total = []
        label_total = []
        for _, (images, labels) in enumerate(test_loader):
            n_example_test += images.size(0)
            images = images.cuda()
            labels = labels.cuda()
            output = net(images)
            avg_loss += nn.CrossEntropyLoss()(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            pred_total.extend(list(pred.data.cpu().numpy()))
            label_total.extend(list(labels.data.cpu().numpy()))
    
    acc = float(total_correct) / n_example_test
    avg_loss /= n_example_test

    # get accuracy per class
    n_class = output.size(1)
    acc_test = [0] * n_class
    cnt_test = [0] * n_class
    for p, l in zip(pred_total, label_total):
        acc_test[l] += int(p == l)
        cnt_test[l] += 1
    acc_per_class = []
    for c in range(n_class):
        acc_test[c] /= float(cnt_test[c])
        acc_per_class.append(acc_test[c])

    return acc, avg_loss.item(), acc_per_class


def get_project_path(ExpID):
    # TODO: change 'project' to 'experiment'
    full_path = glob.glob("Experiments/*%s*" % ExpID)
    assert(len(full_path) == 1) # There should be only ONE folder with <ExpID> in its name.
    return full_path[0]

def check_path(x):
    complete_path = glob.glob(x)
    assert(len(complete_path) == 1)
    x = complete_path[0]
    return x

def mkdirs(*paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

class EMA():
    '''
        Exponential Moving Average for pytorch tensor
    '''
    def __init__(self, mu):
        self.mu = mu
        self.history = {}

    def __call__(self, name, x):
        '''
            Note: this func will modify x directly, no return value.
            x is supposed to be a pytorch tensor.
        '''
        if self.mu > 0:
            assert(0 < self.mu < 1)
            if name in self.history.keys():
                new_average = self.mu * self.history[name] + (1.0 - self.mu) * x.clone()
            else:
                new_average = x.clone()
            self.history[name] = new_average.clone()
            return new_average.clone()
        else:
            return x.clone()

# Exponential Moving Average
class EMA2():
  def __init__(self, mu):
    self.mu = mu
    self.shadow = {}
  def register(self, name, value):
    self.shadow[name] = value.clone()
  def __call__(self, name, x):
    assert name in self.shadow
    new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
    self.shadow[name] = new_average.clone()
    return new_average

def register_ema(emas):
    for net, ema in emas:
        for name, param in net.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

def apply_ema(emas):
    for net, ema in emas:
        for name, param in net.named_parameters():
            if param.requires_grad:
                param.data = ema(name, param.data)

colors = ["gray", "blue", "black", "yellow", "green", "yellowgreen", "gold", "royalblue", "peru", "purple"]
def feat_visualize(ax, feat, label):
    '''
        feat:  N x 2 # 2-d feature, N: number of examples
        label: N x 1
    '''
    for ix in range(len(label)):
        x = feat[ix]
        y = label[ix]
        ax.scatter(x[0], x[1], color=colors[y], marker=".")
    return ax

def smart_weights_load(net, w_path, key=None):
    '''
        This func is to load the weights of <w_path> into <net>.
    '''
    loaded = torch.load(w_path, map_location=lambda storage, location: storage)
    
    # get state_dict
    if isinstance(loaded, collections.OrderedDict):
        state_dict = loaded
    else:
        if key:
            state_dict = loaded[key]
        else:
            if "T" in loaded.keys():
                state_dict = loaded["T"]
            elif "S" in loaded.keys():
                state_dict =  loaded["S"]
            elif "G" in loaded.keys():
                state_dict = loaded["G"]
    
    # remove the "module." surfix if using DataParallel before
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        param_name = k.split("module.")[-1]
        new_state_dict[param_name] = v
    
    # load state_dict into net
    net.load_state_dict(new_state_dict)

    # for name, m in net.named_modules():
    #     module_name = name.split("module.")[-1]
    #     # match and load
    #     matched_param_name = ""
    #     for k in keys_ckpt:
    #       if module_name in k:
    #         matched_param_name = k
    #         break
    #     if matched_param_name:
    #         m.weight.copy_(w[matched_param_name])
    #         print("target param name: '%s' <- '%s' (ckpt param name)" % (name, matched_param_name))
    #     else:
    #         print("Error: cannot find matched param in the loaded weights. please check manually.")
    #         exit(1)

def plot_weights_heatmap(weights, out_path):
    '''
        weights: [N, C, H, W]. Torch tensor
        averaged in dim H, W so that we get a 2-dim color map of size [N, C]
    '''
    w_abs = weights.abs()
    w_abs = w_abs.data.cpu().numpy()
    
    fig, ax = plt.subplots()
    im = ax.imshow(w_abs, cmap='jet')

    # make a beautiful colorbar        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=0.05, pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel("Channel")
    ax.set_ylabel("Filter")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    
    