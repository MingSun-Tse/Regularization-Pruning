"""
This code is from 2019-NIPS-ZSKT (Spotlight): https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/models/wresnet.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.generator as g
import copy


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, out_feat=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8); embed = out
        out = out.view(-1, self.nChannels)
        if out_feat:
          return self.fc(out), embed
        else:
          return self.fc(out)

def ccl(lr_G, lr_S, G_ix, equal_distill=False, embed=False):
    T = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    if equal_distill:
      S = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    else:
      S = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    G = eval("g.Generator" + G_ix)()
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    optim_S = torch.optim.SGD(S.parameters(), lr=lr_S, momentum=0.9, weight_decay=5e-4)
    return T, S, G, optim_G, optim_S

def train_teacher(lr_T, embed=False, student=False):
    T = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    if student:
        T = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    optim_T = torch.optim.SGD(T.parameters(), lr=lr_T, momentum=0.9, weight_decay=5e-4)
    return T, optim_T

def kd(lr_S, equal=False, embed=False):
    T = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    if equal:
        S = copy.deepcopy(T)
    else:
        S = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    optim_S = torch.optim.SGD(S.parameters(), lr=lr_S, momentum=0.9, weight_decay=5e-4)
    return T, S, optim_S

if __name__ == '__main__':
    import random
    import time
    from torchsummary import summary

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    ### WideResNets
    # Notation: W-depth-widening_factor
    #model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=10, dropRate=0.0)
    #model = WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=10, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=1, dropRate=0.0)
    model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    ###model = WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0)

    t0 = time.time()
    output, *act = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHPAE: ", output.shape)

    summary(model, input_size=(3, 32, 32))