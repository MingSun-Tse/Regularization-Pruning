import torch.utils.data as data
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms as transforms
import os
import numpy as np
import torch

def is_img(x):
  _, ext = os.path.splitext(x)
  return ext.lower() in ['.jpg', '.png', '.bmp', '.jpeg']

# Not used in this project.
class CelebA_multi_attr(data.Dataset):
  def __init__(self, img_dir, label_file, transform):
    self.img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if is_img(i)]
    self.transform = transform
    self.label = {}
    num_attributes = 40
    for line in open(label_file):
      if ".jpg" not in line: continue
      img_name, *attr = line.strip().split()
      label = torch.zeros(num_attributes).long()
      for i in range(num_attributes):
        if attr[i] == "1":
          label[i] = 1
      self.label[img_name] = label
  def __getitem__(self, index):
    img_path = self.img_list[index]
    img_name = img_path.split("/")[-1]
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224)) # for alexnet
    img = self.transform(img)
    return img.squeeze(0), self.label[img_name]
  def __len__(self):
    return len(self.img_list)
    
# only for the most balanced attribute "Attractive"
# Deprecated. This class is not fully worked through. Be careful.
class CelebA(data.Dataset):
  def __init__(self, img_dir, label_file, transform):
    self.img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith(".npy")]
    self.transform = transform
    if label_file.endswith(".npy"):
      self.label = np.load(label_file) # label file is npy
    else:
      self.label = {}
      for line in open(label_file): # label file is txt
        if ".jpg" not in line: continue
        img_name, *attr = line.strip().split()
        self.label[img_name] = int(attr[2] == "1") # "Attractive" is at the third position of all attrs
      
  def __getitem__(self, index):
    img_path = self.img_list[index]
    img_name = img_path.split("/")[-1]
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224)) # for alexnet
    img = self.transform(img)
    return img.squeeze(0), self.label[img_name]
  def __len__(self):
    return len(self.img_list)
    
class CelebA_npy(data.Dataset):
  def __init__(self, npy_dir, label_file, transform):
    self.npy_list = [os.path.join(npy_dir, i) for i in os.listdir(npy_dir) if i.endswith(".npy") and i != "batch.npy"]
    self.transform = transform
    self.label = torch.from_numpy(np.load(label_file)).long() # label_file should be an npy
  def __getitem__(self, index):
    npy = self.npy_list[index]
    img = np.load(npy)
    img = Image.fromarray(img)
    img = self.transform(img)
    return img.squeeze(0), self.label[int(npy.split("/")[-1].split(".")[0])]
  def __len__(self):
    return len(self.npy_list)
    
class Dataset_npy_batch(data.Dataset):
  def __init__(self, npy_dir, transform):
    self.data = np.load(os.path.join(npy_dir, "batch.npy"))
    self.transform = transform
  def __getitem__(self, index):
    img = Image.fromarray(self.data[index][0])
    img = self.transform(img)
    label = self.data[index][1]
    label = torch.LongTensor([label])[0]
    return img.squeeze(0), label
  def __len__(self):
    return len(self.data)
