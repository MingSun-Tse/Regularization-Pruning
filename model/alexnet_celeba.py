import torch
import torch.nn as nn
from option import args
import model.generator as g
from torchvision.models import mobilenet_v2, alexnet

class AlexNet(nn.Module):
  def __init__(self, drop=0.5):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      
      nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      
      nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      
      nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      
      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(p=drop),
      nn.Linear(in_features=9216, out_features=4096, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop),
      nn.Linear(in_features=4096, out_features=4096, bias=True),
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=2, bias=True),
    )
  
  def forward(self, x, out_feat=False):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    feat = self.classifier[:6](x) # upto and include ReLU
    x = self.classifier[6:](feat)
    if out_feat:
      return x, feat
    else:
      return x
      
class AlexNet_half(nn.Module):
  def __init__(self, drop=0.5):
    super(AlexNet_half, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      
      nn.Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      
      nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      
      nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      
      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(p=drop),
      nn.Linear(in_features=4608, out_features=2048, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop),
      nn.Linear(in_features=2048, out_features=2048, bias=True),
      nn.ReLU(inplace=True),
      nn.Linear(in_features=2048, out_features=2, bias=True),
    )
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

def ccl(lr_G, lr_S, G_ix, equal_distill=False, embed=False):
    T = AlexNet()
    if equal_distill:
      S = AlexNet()
    else:
      S = AlexNet_half(drop=0.7)
    G = g.Generator6()
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    optim_S = torch.optim.Adam(S.parameters(), lr=lr_S)
    return T, S, G, optim_G, optim_S

def train_teacher(lr_T, embed=False, student=False):
    T = AlexNet()
    if student:
      T = AlexNet_half()
    optim_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    return T, optim_T

def kd(lr_S, equal=False, embed=False):
    T = AlexNet()
    if equal:
      S = AlexNet()
    else:
      S = AlexNet_half()
    optim_S = torch.optim.Adam(S.parameters(), lr=lr_S)
    return T, S, optim_S