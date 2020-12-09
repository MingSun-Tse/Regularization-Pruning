# This file is referring to: [EigenDamage, ICML'19]
# at https://github.com/alecwangcq/EigenDamage-Pytorch

# This file is based on the vgg.py in the same folder with a small change: use two FC layers in the classifier.
# to compare with C-SGD: https://github.com/DingXiaoH/Centripetal-SGD/blob/master/base_model/vgg.py
# they use two FC layers for vgg16.

import math
import torch
import torch.nn as nn
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

_AFFINE = True

defaultcfg = {
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512], # to compare with C-SGD, only vgg16 is needed
    '16_0.5': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256], # to compare with C-SGD, only vgg16 is needed
    '16_0.25': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128], # to compare with C-SGD, only vgg16 is needed
}


class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.features = self.make_layers(cfg, True)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        
        self.classifier = nn.Sequential( # two FC layers for comparison with C-SGD
            nn.Linear(cfg[-1], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
            
        if init_weights:
            self.apply(_weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg19(dataset='cifar100'):
    return VGG(dataset)

def vgg16():
    return VGG(depth='16')

def vgg16_0_5():
    return VGG(depth='16_0.5')

def vgg16_0_25():
    return VGG(depth='16_0.25')