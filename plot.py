import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bbox(img,tar,k):
  img = img.transpose(0,2).transpose(0,1)
  fig = plt.figure(figsize=[20,20])
  ax = fig.add_subplot(111, aspect='equal')
  ax.imshow(img)

  for idx in  range(len(tar['labels'])):
    x, y,xmax,ymax = tar['boxes'][idx]
    if tar['labels'][idx]==0:			######### 0 to 1
      rect = patches.Rectangle((x,y),xmax-x,ymax-y,linewidth=15,edgecolor='r',facecolor='none')
      
    elif tar['labels'][idx]==1:		####### 1 to 2
      rect = patches.Rectangle((x,y),xmax-x,ymax-y,linewidth=15,edgecolor='b',facecolor='none')
      
    else:
      rect = patches.Rectangle((x,y),xmax-x,ymax-y,linewidth=15,edgecolor='k',facecolor='none')
    ax.add_patch(rect)

  #plt.show(block=False)
  plt.savefig(k+'.png')
  #plt.show()
  #plt.pause(5)
  #plt.close()
