import torch
from torchvision import transforms as trf
import numpy as np
import os
from tqdm import tqdm
import torch.optim as optim
from model_arch import create_model
from DataSET import Hist_data
from DataLoader import loader, t_loader
from plot import plot_bbox
from evaluation import conf_mat,make_prediction, f1_score
#from cross_fold import kfold_index, kfold
import time
#from mean_std import find_mean_std


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = "cpu"
print(f"\nDevice = {device}")

"""
8870 0-based Indexing
Hamamatsu_XR = 0-2349
Hamamatsu_S360 = 2350-5645
Aperio = 5646-8869 """

"""
MIDOG'22 Dataset:
41081 
Hamamatsu_XR = 2326 
Hamamatsu_S360 = 5622 
Aperio = 8846 
PanScan = 28892 
Aperio = 36278 
Hamamatsu_XR = 41080"""

#transform = trf.Compose([trf.ToTensor(),
#			  trf.Normalize(mean=[0.7761,0.5452,0.7296],std=[0.1455,0.1933,0.1351])])
transform = trf.ToTensor()
annotation_file = "/home/vaibhav/FuseStyle/Dataset/512_tile.json"

layers = []	# Without FuseStyle

method = input("Press 0 for FuseStyle else press 1 [0/1]=")

if(method!='0'):
	print("\nWithout FuseStyle Implementation\n")
	PATH = '/home/vaibhav/Without_FuseStyle_Weights/'
	
else:
	print("\nFuseStyle Implementation (layer1, layer4)\n")
	layers = ['layer1','layer4']
	PATH = '/home/vaibhav/FuseStyle_Weights/'

print("Select a network for training\n")
print("1.All\n2.XR-CS\n3.S360-CS\n4.XR-S360")
network = int(input("Network = "))
if(network==1):
   newpath = PATH+"All/"
elif(network==2):
   image_dir = "/home1/vaibhav/MIDOG_21_Obj/train/xr-cs/"
   newpath = PATH+"XR_CS/"
   p = 50
elif(network==3):
   image_dir = "/home1/vaibhav/MIDOG_21_Obj/train/s360-cs/"
   newpath = PATH+"S360_CS/"
   p = 100
elif(network==4):
   image_dir = "/home1/vaibhav/MIDOG_21_Obj/train/xr-s360/"
   newpath = PATH+"XR_S360/"
   p = 50

train_ds = Hist_data(image_dir = image_dir ,annotation_file=annotation_file,transform=transform)
xr_test_ds = Hist_data(image_dir = "/home1/vaibhav/MIDOG_21_Obj/test/xr/" ,annotation_file=annotation_file,transform=transform)
s360_test_ds = Hist_data(image_dir = "/home1/vaibhav/MIDOG_21_Obj/test/s360/" ,annotation_file=annotation_file,transform=transform)
cs_test_ds = Hist_data(image_dir = "/home1/vaibhav/MIDOG_21_Obj/test/cs/" ,annotation_file=annotation_file,transform=transform)

a = []
b = []
images = sorted(os.listdir(image_dir))
for i in range(len(images)):
   t = images[i]
   m = int(t[:3])-1
   if m<p:a.append(i)
   else: b.append(i)
   

train_loader = t_loader(train_ds,a=a,b=b)		
XR_loader = loader(xr_test_ds)
S360_loader = loader(s360_test_ds)
CS_loader = loader(cs_test_ds)

## Training XR_CS
model = create_model(2, layers = layers)#['layer1','layer4'])
model = model.to(device)

#params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model.parameters(), lr=1e-4)#, weight_decay=0.1)

def load_state():
  checkpoint = torch.load(newpath+"checkpoint.pth.tar")
  prev_loss = checkpoint['loss']
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  max_f1 = checkpoint['max_f1']
  return prev_loss, max_f1

LOAD = input("Load the model?[y/n]=")
num_epochs = int(input("Number of Epochs="))
prev_loss = np.inf
max_f1 = 0

if (LOAD!="n"):
  if(num_epochs!=0):
    prev_loss,max_f1 = load_state()
    
    print("----------All States Loaded----------")
  
  else:
    #checkpoint = torch.load(newpath+"Best_model.pth.tar")
    if(num_epochs!=0):
       checkpoint = torch.load(newpath+"checkpoint.pth.tar")
    else:
       checkpoint = torch.load(newpath+"Best_model.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("----------Evaluating----------")


# def loss_calculate(model,loader):
#   loss=0
#   with torch.no_grad():
#     for imgs, annotations in loader:
#       imgs = list(img.to(device) for img in imgs)
#       annotation = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#       loss_dict = model(imgs, annotation) 
#       losses = sum(loss for loss in loss_dict.values())
#       loss += losses.item() 
#   return loss                              

# print('\nCalculating losses for current model......')
# XR_loss = loss_calculate(model,XR_loader)
# S360_loss = loss_calculate(model,S360_loader)
# CS_loss = loss_calculate(model,CS_loader)
# Pano_loss = loss_calculate(model,Pano_loader)
# min_loss = XR_loss + S360_loss + CS_loss + Pano_loss
# print(f'Training loss={prev_loss}\n')
# print(f'Validation loss={min_loss}\n')


print('----------------------training started--------------------------')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
f1=0
for epoch in range(num_epochs):
    print(f'Epochs = {epoch+1}/{num_epochs}')
    model.train()
    loop = tqdm(train_loader)
    start = time.time()   
    epoch_loss = 0
    for _,(imgs, annotations) in enumerate(loop):
        imgs = list(img.to(device) for img in imgs)
        annotation = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotation) 
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        epoch_loss += losses.item()
        loop.set_postfix(loss = losses.item())

    print(f'\nTraining loss : {epoch_loss}')
    if(epoch_loss<20):
      # XR_loss = loss_calculate(model,XR_loader)
      # S360_loss = loss_calculate(model,S360_loader)
      # CS_loss = loss_calculate(model,CS_loader)
      # Pano_loss = loss_calculate(model,Pano_loader)
      # test_loss = XR_loss + S360_loss + CS_loss + Pano_loss

      #val_conf = conf_mat(model,val_loader)
      model.eval()
      # train_conf = conf_mat(model,train_loader)
      val_conf_XR = conf_mat(model,XR_loader)
      val_conf_S360 = conf_mat(model,S360_loader)
      val_conf_CS = conf_mat(model,CS_loader)
      
      e_f1_XR = f1_score(val_conf_XR)
      print(f"\nXR F1 Score = {e_f1_XR}\n")
      e_f1_S360 = f1_score(val_conf_S360)
      print(f"\nS360 F1 Score = {e_f1_S360}\n")
      e_f1_CS = f1_score(val_conf_CS)
      print(f"\nCS F1 Score = {e_f1_CS}\n")
      
      train_conf = conf_mat(model,train_loader)
      train_f1 = f1_score(train_conf)
      e_avg_f1 = (e_f1_XR + e_f1_S360 + e_f1_CS)/3

      # print(f'Validation loss={test_loss}\n')

      # print(f'XR Validation loss={XR_loss}\n')
      # print(f'S360 Validation loss={S360_loss}\n')
      # print(f'CS Validation loss={CS_loss}\n')

      if(network==1):f1 = e_avg_f1
      elif(network==2):f1 = e_f1_S360
      elif(network==3):f1 = e_f1_XR
      elif(network==4):f1 = e_f1_CS
      

      print(f'Training F1_score : {train_f1}')
      print(f'Average Validation F1_score : {e_avg_f1}\n')
      
      if(f1>=max_f1):
        print("\n-----Saving Checkpoint-----")
        max_f1 = f1
        print(f"\nTraining Confusion Matrix = \n{train_conf}")
        print(f"\nXR Validation Confusion Matrix = \n{val_conf_XR}")
        print(f"\nS360 Validation Confusion Matrix = \n{val_conf_S360}")
        print(f"\nAperio Validation Confusion Matrix = \n{val_conf_CS}")
        torch.save({
              'model_state_dict': model.state_dict(),
              },newpath+"Best_model.pth.tar")
    
    
    if(epoch_loss>prev_loss):
      epoch_loss, max_f1 = load_state()
      print(f'-------Previous Model Loaded-------')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'max_f1':max_f1
        },newpath+"checkpoint.pth.tar")
    print('--------Saved Checkpoint-------')

    prev_loss = epoch_loss 
    scheduler.step(epoch_loss)
    ##################################################################
    train_loader = t_loader(train_ds,a=a,b=b)
    ##################################################################     
    print(f'time : {time.time() - start}\n')
    if(epoch_loss<10):break
            
print("-----------------------training ended---------------------------")


model.eval()

XR_cm = conf_mat(model,XR_loader)
XR_f1 = f1_score(XR_cm)
print(f"\nXR F1 Score = {XR_f1}\n")

S360_cm = conf_mat(model,S360_loader)
S360_f1 = f1_score(S360_cm)
print(f"\nS360 F1 Score = {S360_f1}\n")

CS_cm = conf_mat(model,CS_loader)
CS_f1 = f1_score(CS_cm)
print(f"\nAperio CS F1 Score = {CS_f1}\n")

i=0
with torch.no_grad(): 
    for imgs, annotations in S360_loader:
      imgs = list(img.to(device) for img in imgs)
      pred = make_prediction(model, imgs, 0.8)
      i+=1
      if(i==27):break


_idx = 2
plot_bbox(imgs[_idx].to("cpu"), annotations[_idx],'1')
plot_bbox(imgs[_idx].to("cpu"), pred[_idx],'1')
#print("Target : ", annotations[_idx])
#print("Prediction : ", pred[_idx])

