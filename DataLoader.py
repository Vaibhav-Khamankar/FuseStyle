from torch.utils.data import DataLoader,ConcatDataset
import random
import torch

def collate_fn(batch):
    return tuple(zip(*batch))

def loader(dataset,batch_size=6):
    return DataLoader(dataset,batch_size=batch_size,collate_fn=collate_fn)


def indexes(a,b):
    random.shuffle(a)
    if(len(b)==0):
        return a
    random.shuffle(b)
    z=[]
    for i in range(max(len(a), len(b))):
        z.append(a[i%len(a)])
        z.append(b[i%len(b)])
    return z

def t_loader(dataset,batch_size=6, a=[], b=[]):
    return DataLoader(torch.utils.data.Subset(dataset,indexes(a,b)),batch_size=batch_size,collate_fn=collate_fn)
