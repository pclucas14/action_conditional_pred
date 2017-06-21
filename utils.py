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
from PIL import Image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def imshow(img, grid=False, epoch=0, display=True):
    # img = img * 0.5
    # img = img + 0.5 
    # img = img * 255. must NOT scale when using pyplot (unlike PIL)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    if npimg.shape[-1] == 3 or grid: 
        plt.imshow(npimg)
    else : 
        plt.imshow(npimg.reshape((npimg.shape[0], npimg.shape[1])))
    if display : 
        plt.show()
    else : 
        plt.savefig('images/'+str(epoch)+'.png')


def show_seq(seq, epoch=0, display=True):
    imshow(torchvision.utils.make_grid(seq), epoch=epoch, grid=True, display=display)

def load_car_data(bbs=500, batch_size=32, seq_len=10, skip=2, big=False):
    filename = 'data/data_driver_filter.bin' if big else 'data/data_driver.bin'
    H, W = (80, 160) if big else (64, 64)
    f = file(filename)
    x = np.load(f)
    # x has shape (7,3) --> 7 recording days, (frame, angle, speed)
    lengths = np.array([xx[0].shape[0] for xx in x]).astype('float32')
    probs = lengths / np.sum(lengths)
    while True : 
        batch = np.zeros((bbs* batch_size, seq_len, 3, H, W))
	for i in range(batch_size * bbs):
	    draw = np.random.multinomial(1, probs, size=1)
	    day_index = np.argmax(draw)
	    # we got our recording. now we pick an image in the day.
	    upper_bound_index = lengths[day_index] - skip * seq_len
	    sample_index = np.random.randint(0, upper_bound_index)
	    sample = x[day_index][0][sample_index:sample_index+ skip * seq_len]
	    batch[i] = sample.transpose(0, 3, 1, 2)[::skip]
        
                
        yield batch / 255.
            
