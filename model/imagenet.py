import torch
import torch.nn as nn
from torchvision.models import alexnet
try:
  from torchvision.models import mobilenet_v2
from option import opt

# modify mobilenet to my interface
class mobilenet_v2_my(nn.Module):
  def __init__(self, pretrained=False):
    super(mobilenet_v2_my, self).__init__()
    if pretrained:
      self.net = mobilenet_v2(True)
    else:
      self.net = mobilenet_v2()
  def forward(self, x, out_feature=False):
    embed = self.net.features(x).mean([2, 3])
    x = self.net.classifier(embed)
    return x, embed if out_feature else x

# modify alexnet to my interface
class alexnet_my(nn.Module):
  def __init__(self, pretrained=False):
    super(alexnet_my, self).__init__()
    if pretrained:
      self.net = alexnet(True)
    else:
      self.net = alexnet()
  def forward(self, x, out_feature=False):
    embed = self.net.features(x).view(x.size(0), -1)
    x = self.net.classifier(embed)
    return x, embed if out_feature else x