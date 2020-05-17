import torch
import torch.nn as nn
import model.generator as g
import copy

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img, out_feat=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feat == False:
            return output
        else:
            return output, feature
    

class LeNet5_half(nn.Module):

    def __init__(self):
        super(LeNet5_half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 42)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(42, 10)

    def forward(self, img, out_feat=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 60)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feat == False:
            return output
        else:
            return output, feature
            
class LeNet5_2neurons(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5_2neurons, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(400, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,   2)
    self.fc6 = nn.Linear(  2,  10)
    self.relu = nn.ReLU(inplace=True)
   
  def forward_embed(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    feat = y.view(y.size(0), -1)
    y = self.relu(self.fc3(feat))
    y = self.relu(self.fc4(y))
    y1 = self.fc5(y)
    y = self.relu(y1)
    y = self.fc6(y)
    return y, y1
  
  def forward(self, y, out_feat=False):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    feat = y.view(y.size(0), -1)
    y = self.relu(self.fc3(feat))
    y = self.relu(self.fc4(y))
    y = self.relu(self.fc5(y))
    y = self.fc6(y)
    if out_feat:
      return y, feat
    else:
      return y

class LeNet5_3neurons(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5_3neurons, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(400, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,   3)
    self.fc6 = nn.Linear(  3,  10)
    self.relu = nn.ReLU(inplace=True)
   
  def forward_embed(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y))
    y = self.relu(self.fc4(y))
    y = self.fc5(y)
    return y
  
  def forward(self, y, out_feat=False):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    feat = y.view(y.size(0), -1)
    y = self.relu(self.fc3(feat))
    y = self.relu(self.fc4(y))
    y = self.relu(self.fc5(y))
    y = self.fc6(y)
    if out_feat:
      return y, feat
    else:
      return y

class LeNet5_bigger(nn.Module):
    def __init__(self):
        super(LeNet5_bigger, self).__init__()

        self.conv1 = nn.Conv2d(1, 12, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(32, 240, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(240, 168)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(168, 10)

    def forward(self, img, out_feat=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 240)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feat == False:
            return output
        else:
            return output, feature

def ccl(lr_G, lr_S, G_ix=1, equal_distill=False, embed=False):
    T = LeNet5_2neurons() if embed else LeNet5()
    if equal_distill:
        S = LeNet5_2neurons() if embed else LeNet5()
    else:
        S = LeNet5_half()
    G = eval("g.Generator" + G_ix)()
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    optim_S = torch.optim.Adam(S.parameters(), lr=lr_S)
    return T, S, G, optim_G, optim_S

def ccl2(lr_G, lr_S, G_ix=1, equal_distill=False, embed=False):
    T = LeNet5_2neurons() if embed else LeNet5()
    T2 = copy.deepcopy(T)
    if equal_distill:
        S = LeNet5_2neurons() if embed else LeNet5()
    else:
        S = LeNet5_half()
    G = eval("g.Generator" + G_ix)()
    G2 = copy.deepcopy(G)
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    optim_G2 = torch.optim.Adam(G2.parameters(), lr=lr_G)
    optim_S = torch.optim.Adam(S.parameters(), lr=lr_S)
    return T, T2, S, G, G2, optim_G, optim_G2, optim_S

def defense(lr_T, lr_G, G_ix=1):
    T = LeNet5()
    G = eval("g.Generator" + G_ix)()
    optim_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    optim_T_partial = torch.optim.Adam(T.fc2.parameters(), lr=lr_T)
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    return T, G, optim_T, optim_G, optim_T_partial

def train_teacher(lr_T, embed=False, student=False):
    T = LeNet5_2neurons() if embed else LeNet5()
    if student:
        T = LeNet5_half()
    optim_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    return T, optim_T

def kd(lr_S, equal=False, embed=False):
    T = LeNet5_2neurons() if embed else LeNet5()
    S = copy.deepcopy(T) if equal else LeNet5_half()
    optim_S = torch.optim.Adam(S.parameters(), lr=lr_S)
    return T, S, optim_S