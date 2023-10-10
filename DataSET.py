import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision import transforms as trf
from PIL import Image
import json

class Hist_data(Dataset):
  def __init__(self,image_dir,annotation_file,transform=None):
    self.image_dir = image_dir
    self.transform = transform
    self.images = sorted(os.listdir(image_dir))
    rows = []
    with open(annotation_file) as f:
        data = json.load(f)
        for row in data:
          rows.append(row)

    self.df = pd.DataFrame(rows, columns=["file_name","x_cord","y_cord","category","scanner"])
    
  def __len__(self):
    return len(self.images)

  def __getitem__(self,index):
    target = {}
    img_path = self.image_dir + '/' + self.images[index] 
    image = Image.open(img_path)
    if self.transform is not None:
      image = self.transform(image)
    image = image[0:3,:,:]
    df = self.df
    out = df.loc[lambda df: df['file_name']==self.images[index], :]
    #box= [[torch.tensor(0),torch.tensor(0),torch.tensor(1),torch.tensor(1)]]
    box = torch.empty(0,4)
    #label = [-1]
    label = torch.empty(0,dtype=torch.int64)
    if len(out)!=0:
      x_cord = torch.tensor(list(out["x_cord"]))
      y_cord = torch.tensor(list(out["y_cord"]))
      box = torch.tensor([list(x) for x in zip(x_cord, y_cord,x_cord+50, y_cord+50)])
      label = torch.tensor(list(out['category']))-1 ###### -1
      if x_cord[0] == -1:
      	box = box = torch.empty(0,4)
      	label = torch.empty(0,dtype=torch.int64)
    target["boxes"] = box
    target["labels"] = label

    #target["boxes"] = torch.tensor(target["boxes"])
    #target["labels"] = torch.tensor(target["labels"])

    return image,target
"""
8870 0-based Indexing
Hamamatsu_XR = 0-2349
Hamamatsu_S360 = 2350-5645
Aperio = 5646-8869 """
#XR_idx = list(range(2327))
#XR_idx.extend(range(36279,41081))
#S360_idx = list(range(2327,5623))
#dataset = torch.utils.data.Subset(dataset,XR_idx)
#dataset = torch.utils.data.Subset(dataset,S360_idx)
