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

class CLSTM_cell(nn.Module):

    def __init__(self, input_shape, output_shape, filter_size):
	super(CLSTM_cell, self).__init__()

	self.input_shape = input_shape
	self.output_shape = output_shape
	padding = (filter_size - 1) / 2
	self.conv = nn.Conv2d(input_shape[1] + output_shape[1], 4 * output_shape[1], filter_size, 1, padding)


    def forward(self, input, hidden_state):
	h_t, c_t = hidden_state
	combined = torch.cat((input, h_t), 1) 
	all_conv = self.conv(combined)
	i, f, o, c_tild = torch.split(all_conv, self.output_shape[1], dim=1)
	
	i = torch.sigmoid(i)
	f = torch.sigmoid(f)
	o = torch.sigmoid(o)
	c_tild = torch.tanh(c_tild)

	next_c = f * c_t + i * c_tild
	next_h = o * torch.tanh(next_c)
	return next_h, next_c


    def init_hidden(self):
	a, b, c, d = self.output_shape
	return (Variable(torch.zeros(a, b, c, d)).cuda(), 
		Variable(torch.zeros(a, b, c, d)).cuda())
    

class CLSTM(nn.Module):
    
    def __init__(self, input_shape, output_shape, filter_size, num_layers):
	super(CLSTM, self).__init__()

	self.input_shape = input_shape
	self.output_shape = output_shape
	self.num_layers = num_layers
	cell_list = [CLSTM_cell(input_shape, output_shape, filter_size).cuda()]

	for _ in range(1, num_layers):
	    cell_list.append(CLSTM_cell(output_shape, output_shape, filter_size).cuda())

        self.cell_list = nn.ModuleList(cell_list)


    '''
    input        : tensor of shape (b_s, seq_len, C_inp, H, W)
    hidden_state : list of shape [(h_1, c_1), ..., (h_n, c_n)] for n layer CLSTM 

    returns
    next_hidden  : list of shape [(h_1, c_1), ..., (h_n, c_n)] for n layer CLSTM
    output       : tensor of shape (b_s, seq_len, C_hid, H, W) 
    '''
    def forward(self, input, hidden_state=None):
	if hidden_state is None : 
	     hidden_state = self.init_hidden()
	 
	input = input.transpose(0,1)
	current_input = input
	next_hidden = []
	seq_len = current_input.size(0)

	for l in range(self.num_layers):
	    h_l, c_l = hidden_state[l]   
	    layer_output = []

	    for t in range(seq_len):
		 h_l, c_l = self.cell_list[l](current_input[t,...], (h_l, c_l))
		 layer_output.append(h_l)

	    # save the last hidden state tuple (for possible hallucination)
	    next_hidden.append((h_l, c_l))            
	    # input of next layer is output of current layer
	    #current_input_old = torch.cat(layer_output, 0).view(current_input.size(0), *layer_output[0].size())
	    current_input = torch.stack(layer_output, 0)         

	return next_hidden, current_input.transpose(0,1)


    def init_hidden(self):
	init_states = []
	for i in range(self.num_layers):
	    init_states.append(self.cell_list[i].init_hidden())
	return init_states
    

############################################################################3
############################################################################3

class TestModel(nn.Module):
    
    def __init__(self, ngpu=1):
        super(TestModel, self).__init__()
        self.ngpu = ngpu
        self.CLSTM = CLSTM((32, 1, 64, 64), (32, 50, 64, 64), 5, 2)
        self.conv = nn.Conv2d(50, 1, 1, padding=0)

    def forward(self, input):
        h_s, out = self.CLSTM(input)
        # let's first try doing the convolution on the last output (not last cell state)
        val = self.conv(out[:, -1, :, :, :])
        # val = nn.ReLU()(val)
        val = torch.sigmoid(val)
        return val


seq_len = 4
clip = 10
batch_size = 32

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
data = data[:, :, :, :, :]
print 'data ready'

trainloader = torch.utils.data.DataLoader(data[:9000], batch_size=batch_size,                                                              shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(data[9000:], batch_size=batch_size, 
                                          shuffle=True, num_workers=2)

opt = optim.Adam(model.parameters(), lr=1e-3)
input = torch.FloatTensor(32, seq_len, 1, 64, 64).cuda()
target = torch.FloatTensor(32, 1, 64, 64).cuda()
criterion = nn.MSELoss()

for epoch in range(500):
    r_loss = 0.
    for i, data in enumerate(trainloader, 0):
        if data.size()[0] != batch_size : continue
        data_l = data
        input.copy_(data[:, :seq_len, :, :, :])
        target.copy_(data[:, seq_len, :, :, :])
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
    imshow(out.cpu().data[0], epoch=epoch, display=False)
    imshow(target_v.cpu().data[0], epoch=1000+epoch, display=False)
    show_seq(data_l[:, :seq_len+1, :, :, :].cpu()[0], epoch=2000+epoch, display=False)
    # if epoch % 5 ==4 : pdb.set_trace()


pdb.set_trace()



