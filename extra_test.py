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


# test model
class TestModel(nn.Module):
    def __init__(self, ngpu=1):
        super(TestModel, self).__init__()
        chan = [32, 64] # channels
        bot = 16 # bottleneck dim
        C = 3
        gru_dim = (32, chan[-1], bot, bot)
        self.ngpu = ngpu
        self.enc = Encoder((32, C, 64, 64), chan)
        self.CGRU1 = CGRU(gru_dim, gru_dim)
        self.CGRU2 = CGRU(gru_dim, gru_dim)
        self.CGRU3 = CGRU(gru_dim, gru_dim)
        self.dec = Decoder((32, C, 64, 64), chan[::-1])
        self.conv = nn.Conv2d(50, C, 1, padding=0)
        self.drop = nn.Dropout2d()


    def forward(self, input):
        val = self.enc(input)
        val = self.CGRU1(val, future=2, return_only_final=False)
        val = self.CGRU2(val, future=2, return_only_final=True)
        val = self.CGRU3(val, future=2, return_only_final=True)
        val = self.dec(val)
        return val


seq_len = 4
clip = 10
batch_size = 32
future = 2
C = 3
model = TestModel(ngpu=1)
model.cuda()
model.apply(weights_init)

generator_train = load_car_data(bbs=100)
generator_test  = load_car_data(bbs=10)
opt = optim.Adam(model.parameters(), lr=1e-3)
input = torch.FloatTensor(32, seq_len, C, 64, 64).cuda()
target = torch.FloatTensor(32, future+1, C, 64, 64).cuda()
criterion = nn.MSELoss()

for epoch in range(500):
    r_loss, t_loss = 0., 0.
    trainloader = torch.utils.data.DataLoader(next(generator_train),
                                    batch_size=batch_size,shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(next(generator_test), 
                                    batch_size=batch_size,shuffle=False, num_workers=2)
    
    for i, data in enumerate(trainloader, 0):
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, seq_len:seq_len+future+1, :, :, :])
        input_v = Variable(input)
        target_v = Variable(target)
        opt.zero_grad()
        out = model(input_v)
        pdb.set_trace()
        loss = criterion(out, target_v)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()
        r_loss += loss.data[0]

    print epoch
    print("%.2f" % r_loss)
    show_seq(out.cpu().data[0], epoch=epoch, display=False)
    show_seq(data[:, :seq_len+future+1, :, :, :].cpu()[0], epoch=2000+epoch, display=False)
    # testing model
    for i, data in enumerate(testloader, 0):
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, seq_len:seq_len+future+1, :, :, :])
        input_v = Variable(input)
        target_v = Variable(target)
        
        opt.zero_grad()
        loss = criterion(model(input_v), target_v)
        loss.backward()
        t_loss += loss.data[0]

    print("%.5f" % t_loss)
    print ""

pdb.set_trace()


