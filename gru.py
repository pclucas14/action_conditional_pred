import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import numpy as np
import pdb
from utils import *
 
class CGRU_cell(nn.Module):

    def __init__(self, input_shape, output_shape, filter_size=7, bn=False):
        super(CGRU_cell, self).__init__()

        # right now code only handles output of same Height / Width
        assert input_shape[0] == output_shape[0]
        assert input_shape[-2:] == output_shape[-2:]
        assert filter_size % 2 == 1
      
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.bn = bn
        padding = (filter_size - 1) / 2

        self.conv_x_to_h = nn.Conv2d(self.input_shape[1], 
                                     3*self.output_shape[1], 
                                     filter_size, 
                                     padding=padding, 
                                     bias=True)

        self.conv_h_to_h = nn.Conv2d(self.output_shape[1], 
                                     2*self.output_shape[1], 
                                     filter_size, 
                                     padding=padding,
                                     bias=False)

        self.conv_s_to_h = nn.Conv2d(self.output_shape[1], 
                                     self.output_shape[1], 
                                     filter_size, 
                                     padding=padding,
                                     bias=False)

        if bn :
            self.bn_z = nn.BatchNorm2d(self.output_shape[1])
            self.bn_r = nn.BatchNorm2d(self.output_shape[1])
            self.bn_h = nn.BatchNorm2d(self.output_shape[1])
 

    def forward(self, input, st=None):
        if st is None : 
            st = self.init_hidden_state()

        # play out one timestep
        x_to_hid = self.conv_x_to_h(input)
        z_x, r_x, h_x = torch.split(x_to_hid, self.output_shape[1], dim=1)
       
        hid_to_hid = self.conv_h_to_h(st)
        z_h, r_h = torch.split(hid_to_hid, self.output_shape[1], dim=1)

        z = z_x + z_h
        r = r_x + r_h

        if self.bn : 
            z = self.bn_z(z)
            r = self.bn_r(r)

        z = nn.Sigmoid()(z)
        r = nn.Sigmoid()(r)

        h_h = self.conv_s_to_h(st * r)
        h = h_x + h_h
        if self.bn : 
            h = self.bn_h(h)

        h = nn.Tanh()(h)
        st = z*h + (1-z)*st

        return st


    def init_hidden_state(self):
        a, b, c, d = self.output_shape
        return Variable(torch.Tensor(a, b, c, d).uniform_(-.08,0.08)).cuda()
                               
        
# class to wrap up cell module : allows you to run on a whole sequence at a time
class CGRU(nn.Module):

    def __init__(self, input_shape, output_shape, bn=False):
        super(CGRU, self).__init__()
        self.cell = CGRU_cell(input_shape, output_shape, bn=bn)

    def forward(self, input, s=None, future=0, return_only_final=True):
        # input is a 5d tensor of shape
        # b_s, seq_len, C, H, W --> seq_len, b_s, C, H, W
        input = torch.transpose(input, 1, 0)
        states = []
        for t in range(input.size()[0]):
            if s is not None : 
                s = self.cell(input[t], s)
            else : 
                s = self.cell(input[t])

            if not return_only_final : states.append(s)

        if future == 0 : 
            if return_only_final : 
                return s
            else : 
                states = torch.stack(states)
                return torch.transpose(states, 1, 0)

        else :
            states = [s] if return_only_final else states 
            # goal : return a 5d tensor with shape b_s, future + 1, C, H, W
            # we use the output (s) as input for the next timestep
            for t in range(future):
                s = self.cell(s, s)
                states.append(s)
             
            return torch.stack(states).transpose(1,0)


# simple encoder structure to reduce input dimensionality
class Encoder(nn.Module):
    # channels : list of filters to use for each convolution (increasing order)
    # every layer uses stride 2 and divides dim / 2. 
    def __init__(self, input_shape, channels, filter_size=4, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        assert filter_size % 2 == 0
        self.input_shape = input_shape
        padding = filter_size / 2 - 1
        self.convs = [nn.Conv2d(input_shape[1], channels[0], filter_size, stride=2, padding=padding)]
        for i in range(1, len(channels)):
            self.convs.append(nn.Conv2d(channels[i-1], channels[i], filter_size, stride=2, padding=padding))
        
        self.convs = nn.ModuleList(self.convs)
        self.activation = activation

    def forward(self, input):
        sh = input.size()
        shrink = 2 ** len(self.convs)
        val = input.contiguous().view((sh[0]*sh[1], sh[2], sh[3], sh[4])) if len(sh) == 5 else input
        for i in range(len(self.convs)):
             val = self.convs[i](val)
             val = self.activation(val)
        
        val = val.view(sh[0], sh[1], -1, sh[3] / shrink, sh[4] / shrink) if len(sh) == 5 else val
        return val
        

# simple decoder structure to project back to original dimensions
class Decoder(nn.Module):
    # channels : list of filters to use for each convolution (DECreasing order)
    # every layer uses stride 2 and divides dim / 2. 
    def __init__(self, output_shape, channels, filter_size=4, activation=nn.ReLU()):
        super(Decoder, self).__init__()
        assert filter_size % 2 == 0
        self.output_shape = output_shape
        padding = filter_size / 2 - 1
        self.deconvs = []
        for i in range(len(channels)-1):
            self.deconvs.append(nn.ConvTranspose2d(channels[i], channels[i+1], filter_size, stride=2, padding=padding))
        
        self.deconvs.append(nn.ConvTranspose2d(channels[-1], output_shape[1], filter_size, stride=2, padding=padding))
        self.deconvs = nn.ModuleList(self.deconvs)
        self.activation = activation

    def forward(self, input):
        sh = input.size()
        shrink = 2 ** len(self.deconvs)
        val = input.contiguous().view((sh[0]*sh[1], sh[2], sh[3], sh[4])) if len(sh) == 5 else input
        for i in range(len(self.deconvs)):
             val = self.deconvs[i](val)
             val = self.activation(val)

        val = val.view(sh[0], sh[1], -1, sh[3] * shrink, sh[4] * shrink) if len(sh) == 5 else val
        return val


# test model
class TestModel(nn.Module):
    
    def __init__(self, ngpu=1):
        super(TestModel, self).__init__()
        self.ngpu = ngpu
        self.enc = Encoder((32, 1, 64, 64), [32, 64])
        self.CGRU1 = CGRU((32, 64, 16, 16), (32, 64, 16, 16))
        self.CGRU2 = CGRU((32, 64, 16, 16), (32, 64, 16, 16))
        self.dec = Decoder((32, 1, 64, 64), [64, 32])
        self.conv = nn.Conv2d(50, 1, 1, padding=0)
        self.drop = nn.Dropout2d()

    def forward(self, input):
        val = self.enc(input)
        val = self.CGRU1(val, future=2, return_only_final=False)
        val = self.CGRU2(val, future=2, return_only_final=True)
        val = self.dec(val)
        return val

seq_len = 2
clip = 10
batch_size = 32
future = 2

model = TestModel(ngpu=1)
model.cuda()
model.apply(weights_init)
print 'loading data'
f = file('bouncing_mnist_test.npy', 'rb')
#f = file('data_driver_filter_sqr.bin')
data = np.load(f).astype('float32')
#for day in range(data.shape[0]):
   

data /= 255.; #data -= 0.5; data /= 0.5
data = data.reshape((10000, 20, 1, 64, 64))
data = data[:, ::3, :, :, :]
print 'data ready'
trainloader = torch.utils.data.DataLoader(data[:9000], batch_size=batch_size,                                                              shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(data[9000:], batch_size=batch_size, 
                                          shuffle=True, num_workers=2)

opt = optim.Adam(model.parameters(), lr=1e-3)
input = torch.FloatTensor(32, seq_len, 1, 64, 64).cuda()
target = torch.FloatTensor(32, future+1, 1, 64, 64).cuda()
criterion = nn.MSELoss()

for epoch in range(500):
    r_loss = 0.
    for i, data in enumerate(trainloader, 0):
        if data.size()[0] != batch_size : continue
        data_l = data
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, seq_len:seq_len+future+1, :, :, :])
        input_v = Variable(input)
        target_v = Variable(target)

        opt.zero_grad()
        out = model(input_v)
        loss = criterion(out, target_v)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()
        r_loss += loss.data[0]
    print epoch
    print("%.2f" % r_loss)
    print ""
    # imshow(out.cpu().data[0], epoch=epoch, display=False)
    # imshow(target_v.cpu().data[0], epoch=1000+epoch, display=False)
    show_seq(out.cpu().data[0], epoch=epoch, display=False)
    show_seq(data_l[:, :seq_len+future+1, :, :, :].cpu()[0], epoch=2000+epoch, display=False)
    # if epoch % 5 ==4 : pdb.set_trace()


pdb.set_trace()


