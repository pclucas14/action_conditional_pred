import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import pdb
import torch.autograd as autograd


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def imshow(img, grid=False, epoch=0, display=True):
    img = img * 0.5
    # img = img + 0.5 
    # img = img * 255.
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    if npimg.shape[0] == 3 or grid: 
        plt.imshow(npimg)
    else : 
        plt.imshow(npimg.reshape((npimg.shape[0], npimg.shape[1])))
    if display : 
        plt.show()
    else : 
        plt.savefig('images/'+str(epoch)+'.png')

def show_seq(seq, epoch=0, display=True):
    imshow(torchvision.utils.make_grid(seq), epoch=epoch, grid=True, display=display)

