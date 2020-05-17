import torch
import torch.nn as nn
from torchvision.models import alexnet
try:
  from torchvision.models import mobilenet_v2
except:
  pass
from model import generator as g

# modify mobilenet to my interface
class MobilenetV2(nn.Module):
  def __init__(self, n_class=1000, width_mult=1.0):
    super(MobilenetV2, self).__init__()
    self.net = mobilenet_v2(width_mult=width_mult)
    self.net.classifier = nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=1280, out_features=n_class, bias=True)
    )

  def forward(self, x, out_feat=False):
    embed = self.net.features(x).mean([2, 3])
    x = self.net.classifier(embed)
    return (x, embed) if out_feat else x

# modify alexnet to my interface
class AlexNet(nn.Module):
  def __init__(self, pretrained=False):
    super(AlexNet, self).__init__()
    if pretrained:
      self.net = alexnet(True)
    else:
      self.net = alexnet()
  def forward(self, x, out_feat=False):
    embed = self.net.features(x).view(x.size(0), -1)
    x = self.net.classifier(embed)
    return (x, embed) if out_feat else x