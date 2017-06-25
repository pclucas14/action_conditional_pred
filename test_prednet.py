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
from models import * 

load_weights = True
seq_len = 6
batch_size = 20
future = 5
C, H, W = 3, 64, 128
big=1
extra = True

input_shape = (batch_size, 3, H, W) 
A_channels = (C, 48, 96, 192)
R_channels = A_channels
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
clip=100

target_index = seq_len if future == 0 else slice(seq_len,seq_len+future+1)
test_index = slice(0,seq_len+1) if future == 0 else slice(0, seq_len+future+1)
tensor_size = 1 if future == 0 else future + 1
model = PredNet(input_shape, A_channels, R_channels, A_filt_sizes, 
                Ahat_filt_sizes, R_filt_sizes, extra=True)
model.cuda()
if load_weights : 
    model.load_state_dict(torch.load('models/exp4/model.pth'))
    print 'PredNet weights loaded'
else : 
    model.apply(weights_init)

generator_train = load_car_data(bbs=100, skip=2, seq_len=15, big=big)
opt = optim.Adam(model.parameters(), lr=1e-4)
input = torch.FloatTensor(batch_size, seq_len, C, H, W).cuda()
target = torch.FloatTensor(batch_size, tensor_size, C, H, W).cuda()
extra = torch.FloatTensor(batch_size, 15, 15).cuda()
criterion = nn.MSELoss()

for epoch in range(1000):
    print epoch
    r_loss, t_loss = 0., 0.
    data_train= next(generator_train)
    trainloader = torch.utils.data.DataLoader(data_train[0],
                                    batch_size=batch_size,shuffle=False, num_workers=2)
    extraloader = torch.utils.data.DataLoader(data_train[1], 
                                    batch_size=batch_size,shuffle=False, num_workers=2)
    
    for i, data_all in enumerate(zip(trainloader, extraloader), 0):
        data, extra_d = data_all
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, target_index, :, :, :])
        extra.copy_(extra_d)
        input_v = Variable(input)
        target_v = Variable(target)
        extra_v = Variable(extra)
        opt.zero_grad()
        out = model(input_v, future=future, extra=extra_v)
        loss = criterion(out, target_v)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()
        r_loss += loss.data[0]

    
    print("%.2f" % r_loss)
    print ""

    # try with random extra values
    extra.normal_()
    extra_v = Variable(extra)
    out_corrupt = model(input_v, future=future, extra=extra_v)
    show_seq(out.cpu().data[0], epoch=epoch, display=False)
    show_seq(out_corrupt.cpu().data[0], epoch=1000+epoch, display=False)
    show_seq(data[:, test_index, :, :, :].cpu()[0], epoch=2000+epoch, display=False)

    if epoch + 1 % 100 == 0 : pdb.set_trace()

pdb.set_trace()


