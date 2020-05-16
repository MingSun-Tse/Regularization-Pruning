import torch
import torch.nn as nn
from option import args
import model.generator as g

# ref to ZSKD: https://github.com/vcl-iisc/ZSKD/blob/master/model_training/include/model_alex_full.py
class AlexNet_cifar10(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(AlexNet_cifar10, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.LocalResponseNorm(size=2),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.BatchNorm2d(48),

      nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.LocalResponseNorm(size=2),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.BatchNorm2d(128),

      nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(192),

      nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(192),

      nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.BatchNorm2d(128), # output: 128x3x3
    )
    self.classifier = nn.Sequential(
      nn.Linear(in_features=1152, out_features=512, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      # nn.BatchNorm2d(512),
      
      nn.Linear(in_features=512, out_features=256, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      # nn.BatchNorm2d(256),
      
      nn.Linear(in_features=256, out_features=10, bias=True),
    )
    self.fc_bn1 = nn.BatchNorm2d(512).cuda()
    self.fc_bn2 = nn.BatchNorm2d(256).cuda()
    
  def forward(self, x, out_feat=False):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier[ :3](x); x = x.view(x.size(0), -1, 1, 1); x = self.fc_bn1(x); x = x.view(x.size(0), -1)
    x = self.classifier[3:6](x); x = x.view(x.size(0), -1, 1, 1); f = self.fc_bn2(x); f = f.view(f.size(0), -1)
    x = self.classifier[  6](f)
    if out_feat:
      return x, f
    else:
      return x
      
# ref to ZSKD: https://github.com/vcl-iisc/ZSKD/blob/master/model_training/include/model_alex_half.py
class AlexNet_cifar10_student(nn.Module):
  def __init__(self, model=None, fixed=False, drop=0.5):
    super(AlexNet_cifar10_student, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.LocalResponseNorm(size=2),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.BatchNorm2d(24),

      nn.Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
      nn.ReLU(inplace=True),
      nn.LocalResponseNorm(size=2),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.BatchNorm2d(64),

      nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(96),

      nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(96),

      nn.Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.BatchNorm2d(64), # output: 128x3x3
    )
    self.classifier = nn.Sequential(
      nn.Linear(in_features=576, out_features=256, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop),
      # nn.BatchNorm2d(256),
      
      nn.Linear(in_features=256, out_features=128, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop),
      # nn.BatchNorm2d(128),
      
      nn.Linear(in_features=128, out_features=10, bias=True),
    )
    self.fc_bn1 = nn.BatchNorm2d(256).cuda()
    self.fc_bn2 = nn.BatchNorm2d(128).cuda()
  
  def forward(self, x, out_feat=False):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier[ :3](x); x = x.view(x.size(0), -1, 1, 1); x = self.fc_bn1(x); x = x.view(x.size(0), -1)
    x = self.classifier[3:6](x); x = x.view(x.size(0), -1, 1, 1); f = self.fc_bn2(x); f = f.view(f.size(0), -1)
    x = self.classifier[  6](f)
    if out_feat:
      return x, f
    else:
      return x

def ccl(lr_G, lr_S, G_ix, equal_distill=False, embed=False):
    T = AlexNet_cifar10()
    if equal_distill:
      S = AlexNet_cifar10()
    else:
      S = AlexNet_cifar10_student()
    G = eval("g.Generator" + G_ix)()
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    optim_S = torch.optim.SGD(S.parameters(), lr=lr_S, momentum=0.9, weight_decay=5e-4)
    return T, S, G, optim_G, optim_S

def train_teacher(lr_T, embed=False, student=False):
    T = AlexNet_cifar10()
    if student:
        T = AlexNet_cifar10_student()
    optim_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    return T, optim_T

def kd(lr_S, equal=False, embed=False):
    T = AlexNet_cifar10()
    S = AlexNet_cifar10() if equal else AlexNet_cifar10_student()
    optim_S = torch.optim.Adam(S.parameters(), lr=lr_S)
    return T, S, optim_S