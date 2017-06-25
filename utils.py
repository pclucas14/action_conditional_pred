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

def load_car_data(bbs=100, batch_size=32, seq_len=10, skip=2, big=2):
    # all_ = np.stack((frames, speeds, angles, velos, accs), axis=1)
    if big == 2 : 
        filename = 'data/data_80x160_extraf.bin' #'data/data_driver_filter.bin'
        H, W = 80, 160
    elif big == 1: 
        H, W = 64, 128
        filename = 'data/data_64x128_extraf.bin' #'data/data_driver_allf.bin'
    elif big == 0 : 
        H, W = 64, 64
        filename = 'data/data_driver.bin'

    f = file(filename)
    x = np.load(f)
    lengths = np.array([xx[0].shape[0] for xx in x]).astype('float32')
    probs = lengths / np.sum(lengths)
    ang_mean = np.array([np.mean(xx[1]) for xx in x]).astype('float32')
    ang_std = np.array([np.std(xx[1]) for xx in x]).astype('float32')
    speed_mean = np.array([np.mean(xx[2]) for xx in x]).astype('float32')
    speed_std = np.array([np.std(xx[2]) for xx in x]).astype('float32')
    velo_mean = np.array([np.mean(xx[3]) for xx in x]).astype('float32')
    velo_std = np.array([np.std(xx[3]) for xx in x]).astype('float32')
    acc_mean = np.array([np.mean(xx[4]) for xx in x]).astype('float32')
    acc_std = np.array([np.std(xx[4]) for xx in x]).astype('float32')

    ang_mean = (probs * ang_mean).sum()
    ang_std = (probs * ang_std).sum()
    speed_mean = (probs * speed_mean).sum()
    speed_std = (probs * speed_std).sum()
    velo_mean = (probs * velo_mean).sum()
    velo_std = (probs * velo_std).sum()
    acc_mean = (probs * acc_mean).sum()
    acc_std = (probs * acc_std).sum()

    while True : 
        batch = np.zeros((bbs * batch_size, seq_len, 3, H, W))
        speed = np.zeros((bbs * batch_size, seq_len))
        angle = np.zeros((bbs * batch_size, seq_len))
        acc   = np.zeros((bbs * batch_size, seq_len))
        velo  = np.zeros((bbs * batch_size, seq_len, 12))
	for i in range(batch_size * bbs):
	    draw = np.random.multinomial(1, probs, size=1)
	    day_index = np.argmax(draw)

	    # we got our recording. now we pick an image in the day.
	    upper_bound_index = lengths[day_index] - skip * seq_len
	    sample_index = np.random.randint(0, upper_bound_index)
	    sample = x[day_index][0][sample_index:sample_index + skip * seq_len]
	    batch[i] = sample.transpose(0, 3, 1, 2)[::skip]
            angle[i] = x[day_index][1][sample_index:sample_index + skip * seq_len][::skip]
            speed[i] = x[day_index][2][sample_index:sample_index + skip * seq_len][::skip]
            velo[i]  = x[day_index][3][sample_index:sample_index + skip * seq_len][::skip]
            acc[i]   = x[day_index][4][sample_index:sample_index + skip * seq_len][::skip]
              
        speed = (speed - speed_mean) / speed_std
        angle = (angle - ang_mean) / ang_std
        velo  = (velo  - velo_mean) / velo_std
        acc   = (acc   - acc_mean) / acc_std
        
        speed = speed.reshape((-1, seq_len, 1))
        angle = angle.reshape((-1, seq_len, 1))
        acc   =   acc.reshape((-1, seq_len, 1))

        extra = np.concatenate((speed, angle, acc, velo), axis=2)

        yield batch / 255., extra.astype('float32')
            
