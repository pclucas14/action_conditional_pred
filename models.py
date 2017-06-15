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
    def __init__(self, input_shape, output_shape, bn=True):
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

