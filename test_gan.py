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

BN = False
load = False

# test model
class Gen(nn.Module):
    def __init__(self, ngpu=1):
        super(Gen, self).__init__()
        chan = [32, 64] # channels
        bot = 16 # bottleneck dim
        bs = 32
        C = 3
        fs = 7
        gru_dim = (bs, chan[-1], bot, bot)
        self.enc = Encoder((32, C, 64, 64), chan, bn=BN)
        self.CGRU1 = CGRU(gru_dim, gru_dim, bn=BN)
        '''
        self.CGRU2 = CGRU(gru_dim, (bs, 64, 16, 16), filter_size=fs, bn=BN)
        self.CGRU3 = CGRU((bs, 64, 16, 16), (bs, 32, 32, 32), filter_size=fs, bn=BN)
        self.CGRU4 = CGRU((bs, 32, 32, 32), (bs, C, 64, 64), filter_size=fs, bn=BN)
        '''
        self.CGRU2 = CGRU(gru_dim, gru_dim, bn=BN)
        # self.CGRU3 = CGRU(gru_dim, gru_dim, bn=BN)
        self.dec = Decoder((32, C, 64, 64), chan[::-1], bn=BN)
        self.conv = nn.Conv2d(50, C, 1, padding=0)
        

    def forward(self, input):
        val = self.enc(input)
        val = self.CGRU1(val, future=4, return_only_final=False)
        # TODO : put this back to false
        val = self.CGRU2(val, future=0, return_only_final=False)
        # val = self.CGRU3(val, future=0, return_only_final=False)
        #val = self.CGRU4(val, future=0, return_only_final=False)
        val = self.dec(val[:, -5:, :, :, :])
        # val = self.dec(val)
        return val#[:, -5:, :, :, :]



# test model
class Disc(nn.Module):
    def __init__(self, ngpu=1):
        super(Disc, self).__init__()
        chan = [32, 64] # channels
        bot = 16 # bottleneck dim
        C = 3
        gru_dim = (32, chan[-1], bot, bot)
        self.ngpu = ngpu
        self.enc = Encoder((32, C, 64, 64), chan, bn=BN)
        self.CGRU1 = CGRU(gru_dim, gru_dim, bn=BN)
        self.CGRU2 = CGRU(gru_dim, gru_dim, bn=BN)
        # self.CGRU3 = CGRU(gru_dim, gru_dim, bn=BN)
        chan = [64, 128]
        self.enc_end = Encoder((32, chan[0], bot, bot), chan, bn=BN, last_nl=None)


    def forward(self, input):
        val = self.enc(input)
        val = self.CGRU1(val, return_only_final=False)
        hid = self.CGRU2(val, return_only_final=False)
        # val = self.CGRU3(hid, return_only_final=True)
        # TODO : only encode last state ?
        val = self.enc_end(hid)
        return hid, val.squeeze(2).squeeze(2) # b_s, 1, 1, 1 ==> b_s, 1



seq_len = 3
clip = 10
batch_size = 32
future = 4
C = 3
disc_iter, gen_iter = 2, 1
lambda_mse = 0
lambda_adv = 1
lambda_hid = 0

netG = Gen()
netD = Disc()
netD.cuda(); netG.cuda()
netD.apply(weights_init); netG.apply(weights_init)
if load : 
    netD.load_state_dict(torch.load('netD_epoch_999.pth'))
    netG.load_state_dict(torch.load('netG_epoch_999.pth'))

generator_train = load_car_data(bbs=50, skip=5)
generator_test  = load_car_data(bbs=3, skip=5)
optG = optim.RMSprop(netG.parameters(), lr=1e-4)
optD = optim.RMSprop(netD.parameters(), lr=1e-4)

input = torch.FloatTensor(32, seq_len, C, 64, 64).cuda()
target = torch.FloatTensor(32, future+1, C, 64, 64).cuda()

for epoch in range(10000):
    trainloader = torch.utils.data.DataLoader(next(generator_train),
                                    batch_size=batch_size,shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(next(generator_test), 
                                    batch_size=batch_size,shuffle=False, num_workers=2)
   
    real_outs, fake_outs, t_loss = 0, 0, 0
    for i, data in enumerate(trainloader, 0):
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, seq_len:seq_len+future+1, :, :, :])
        input_v = Variable(input)
        target_v = Variable(target)
    
        '''
        discriminator pass
        '''
        for _ in range(disc_iter):
            netD.zero_grad()
            # train on fake data
            gen_seq = netG(input_v)
            _, fake_out = netD(gen_seq.detach())
            fake_outs += fake_out.mean()

            # train on real data
            real_hid, real_out = netD(target_v)
            real_outs += real_out.mean()

            D_loss = 0.5 * (torch.mean((real_out - 1) ** 2) + torch.mean(fake_out ** 2))
            D_loss.backward()
            optD.step()

        '''
        generator pass
        '''
        for _ in range(gen_iter):
            netG.zero_grad()
            # generate fake data 
            gen_seq = netG(input_v)
            fake_hid, fake_out = netD(gen_seq)
            G_loss_adv = 0.5 * torch.mean((fake_out -1) ** 2)
            G_loss_mse = torch.mean((gen_seq - target_v) ** 2)
            G_loss_hid = torch.mean((fake_hid - real_hid.detach()+ 1e-8) ** 2)
            G_loss = lambda_mse * G_loss_mse + lambda_adv * G_loss_adv + lambda_hid * G_loss_hid
            G_loss.backward()
            optG.step()

    print epoch
    print("%.5f" % (real_outs.data[0] / (i * disc_iter)))
    print("%.5f" % (fake_outs.data[0] / (i * disc_iter)))
    show_seq(gen_seq.cpu().data[0], epoch=epoch, display=False)
    show_seq(data[:, :seq_len+future+1, :, :, :].cpu()[0], epoch=2000+epoch, display=False)
    
    # testing model
    for i, data in enumerate(testloader, 0):
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, seq_len:seq_len+future+1, :, :, :])
        input_v = Variable(input)
        target_v = Variable(target)
        
        gen_seq = netG(input_v)
        loss = torch.mean((gen_seq - target_v) ** 2)
        t_loss += loss.data[0]

    print("%.5f" % t_loss)
    print ""

pdb.set_trace()


