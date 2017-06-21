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
batch_size = 32
future = 5
C = 3

input_shape = (32, 3, 80, 160) # 64, 64)
A_channels = (C, 48, 96, 192)
R_channels = A_channels
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
clip=100
target_index = seq_len if future == 0 else slice(seq_len,seq_len+future+1)
test_index = slice(0,seq_len+1) if future == 0 else slice(0, seq_len+future+1)
tensor_size = 1 if future == 0 else future + 1
model = PredNet(input_shape, A_channels, R_channels, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes)
model.cuda()
if load_weights : 
    model.load_state_dict(torch.load('models/prednet_single_frame.pth'))
    print 'PredNet weights loaded'
else : 
    model.apply(weights_init)

# generator_train = load_car_data(bbs=100, skip=2, seq_len=15, big=True)
generator_test  = load_car_data(bbs=10, skip=2, seq_len=15, big=True)
opt = optim.Adam(model.parameters(), lr=2e-4)
input = torch.FloatTensor(32, seq_len, C, 80, 160).cuda()
# target = torch.FloatTensor(32, tensor_size, C, 80, 160).cuda()
criterion = nn.MSELoss()

for epoch in range(500):
    print epoch
    r_loss, t_loss = 0., 0.
    #trainloader = torch.utils.data.DataLoader(next(generator_train),
    #                                batch_size=batch_size,shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(next(generator_test), 
                                    batch_size=batch_size,shuffle=False, num_workers=2)
    ''' 
    for i, data in enumerate(trainloader, 0):
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, target_index, :, :, :])
        input_v = Variable(input)
        target_v = Variable(target)
        opt.zero_grad()
        out = model(input_v, future=future)
        loss = criterion(out, target_v)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()
        r_loss += loss.data[0]

    
    print("%.2f" % r_loss)
    show_seq(out.cpu().data[0], epoch=epoch, display=False)
    show_seq(data[:, test_index, :, :, :].cpu()[0], epoch=2000+epoch, display=False)
    '''

    # testing model
    for i, data in enumerate(testloader, 0):
        input.copy_(data[:, :seq_len, :, :, :])
        # target.copy_(data[:, target_index, :, :, :])
        input_v = Variable(input)
        # target_v = Variable(target)
        
        # opt.zero_grad()
        out = model(input_v, future=6)
        # loss = criterion(out, target_v)
        # loss.backward()
        # t_loss += loss.data[0]
        pdb.set_trace()
        

    print("%.5f" % t_loss)
    print ""

pdb.set_trace()


