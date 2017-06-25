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


ngpu = 1 
nz = 100
ngf = 64
ndf = 64
nc = 3
gpu1, gpu2 = 0, 1
load_weights = True

'''
Taken from pytorch's DCGAN repo : 
https://github.com/pytorch/examples/blob/master/dcgan/main.py
'''
class _netG(nn.Module):
    def __init__(self, ngpu, encode=False):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        if encode : 
            self.main = nn.Sequential(
                # part 1 : encoder 
                # input is nc x 80 x 160
                nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
                nn.ReLU(True),
                # state size. (ngf) x 40 x 80
                nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2), 
                nn.ReLU(True),
                # state size. (ngf*2) x 20 x 40
                nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False), 
                nn.BatchNorm2d(ngf * 4), 
                nn.ReLU(True),
                # state size. (ngf*4) x 10 x 20
                nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False), 
                nn.BatchNorm2d(ngf * 8), 
                nn.ReLU(True), 
                # state size. (ngf*8) x 5 x 10 

                # part 2 : decoder
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 10 x 20
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 20 x 40
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 40 x 80
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                # nn.Tanh()
                nn.Sigmoid()
                # state size. (nc) x 80 x 160
            )
        else : 
            self.main = nn.Sequential(
                    nn.Conv2d(nc, 64, 3, 1, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True), 
                    nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(64), 
                    nn.LeakyReLU(0.2, inplace=True), 
                    nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(64), 
                    nn.LeakyReLU(0.2, inplace=True), 
                    nn.Conv2d(64, nc, 3, 1, 1, bias=False),
                    nn.Sigmoid()
            )
            

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu, encode=True)
netG.apply(weights_init)
print(netG)

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)# ,
            # nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)


netD = _netD(ngpu)
netD.apply(weights_init)
print(netD)

# optimizers 
optimizerD = optim.RMSprop(netD.parameters(), lr=2e-4)
optimizerG = optim.RMSprop(netG.parameters(), lr=2e-4)

seq_len = 6
batch_size = 64
future = 5
C, H, W = 3, 80, 160
lambda_mse = 50
lambda_adv = 1

# prednet parameters
input_shape = (batch_size, 3, 80, 160)
A_channels = (C, 48, 96, 192)
R_channels = A_channels
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

'''
model = PredNet(input_shape, A_channels, R_channels, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes)
if load_weights : 
    model.load_state_dict(torch.load('models/prednet_4_frames.pth'))
    print 'PredNet weights loaded'
model.cuda(gpu1); 
'''

netD.cuda(); netG.cuda()

generator_train = load_car_data(bbs=300, seq_len=12, skip=2, big=2)
input_pn = torch.FloatTensor(batch_size, seq_len, C, H, W).cuda() #gpu1
target_pn = torch.FloatTensor(batch_size, C, H, W).cuda()
blurred = torch.FloatTensor(batch_size, C, H, W).cuda()

for epoch in range(500):
    trainloader = torch.utils.data.DataLoader(next(generator_train)[0],
                        batch_size=batch_size,shuffle=False, num_workers=2)
    real, fake_g, fake_d = 0, 0, 0

    for i, data in enumerate(trainloader, 0):
        # run the sequence through prednet to get -blurred- output
        input_pn.copy_(data[:, :seq_len, :, :, :])
        target_pn.copy_(data[:, seq_len + future, :, :, :])
        input_v = Variable(input_pn)
        target_v = Variable(target_pn)
        # out_pn = model(input_v, future=future, return_only_final=True)
        blurred.copy_(data.mean(1))
        out_pn = Variable(blurred)
        # out_pn = out_pn.cuda(gpu2)
        fake = netG(out_pn.detach())
  
        '''
        Update D network 
        '''
        # train with real data
        netD.zero_grad()
        real_out = netD(target_v)
        loss_real = 0.5 * torch.mean((real_out - 1) ** 2)
        real += real_out.mean()
        loss_real.backward()

        # train with fake data
        fake_out = netD(fake.detach())
        loss_fake = 0.5 * torch.mean((fake_out - 0) ** 2)
        fake_d += fake_out.mean()
        loss_fake.backward()
        optimizerD.step()

        '''
        Update G network
        '''
        netG.zero_grad()
        fake_out = netD(fake)
        loss_adv = 0.5 * torch.mean((fake_out - 1) ** 2)
        # loss_mse = torch.mean((out_pn - fake) ** 2)
        loss_mse = torch.mean((out_pn - fake) ** 2)
        fake_g += fake_out.mean()
        loss = lambda_adv * loss_adv + lambda_mse * loss_mse
        loss.backward()
        optimizerG.step()        

    print epoch
    print("%.5f" % (real.data[0] / i))
    print("%.5f" % (fake_d.data[0] / i))
    print("%.5f" % (fake_g.data[0] / i))
    show_seq(fake.cpu().data[0], epoch=epoch, display=False)
    show_seq(target_v.cpu().data[0], epoch=1000+epoch, display=False)
    show_seq(out_pn.cpu().data[0], epoch=2000+epoch, display=False)
    print ""

pdb.set_trace()


