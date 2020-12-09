'''
    implement the A-softmax loss from
    <SphereFace: Deep Hypersphere Embedding for Face Recognition>
    @author: Yuan Yang
    @date: 2017.05.25
'''

import math
from easydict import EasyDict as edict
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import Function
import torch
import numpy as np


class guli_fun(Function):
    '''
    actual computing of a-softmax loss happens here
    '''
    def __init__(self, beta=10.0):
        '''
            get params from SphereLoss module
        '''
        self.c_map = list()
        self.k_map = list()
        self.margin = 4

        # look up table for C_m_n and cos_mt
        c_m_n = lambda m, n: math.factorial(n) / math.factorial(m) / math.factorial(n-m)
        for i in range(self.margin+1):
            self.c_map.append(c_m_n(i, self.margin))
            self.k_map.append(math.cos(i*math.pi / self.margin))

        # save them in forward operation, used in backward
        self.k = None
        self.cos_t = None
        self.cos_mt = None
        self.x_norm = None
        self.is_cuda = False

        # eps to avoid nan in division
        self.eps = 1e-6

        self.beta = beta

    def find_k(self, cos_t):
        '''
            find k for cos(theta)
            cos_t is a scalar
        '''
        # for numeric issue
        eps = 1e-5
        le = lambda x, y: x < y or abs(x-y) < eps
        for i in range(self.margin):
            if le(self.k_map[i+1], cos_t) and le(cos_t, self.k_map[i]):
                return i
        raise ValueError('can not find k for cos_t = %f'%clamp_cos)

    def calc_cos_mt(self, cos_t):
        '''
            calculate cos(m*theta)
            cos_t is vector
        '''
        clamp_cos = torch.clamp(cos_t, -1.0, 1.0)
        cos_mt = torch.cos(self.margin*torch.acos(clamp_cos))
        return cos_mt


    def forward(self, input, weight, label):
        self.is_cuda = input.is_cuda
        self.save_for_backward(input, label)
        
        self.weight = weight

        # original Linear operation
        out = torch.mm(input, self.weight.t()) # n x self.out_f

        self.x_norm = torch.norm(input, 2, 1)

        x_dot_w_yi = torch.gather(out, 1, label.view(label.size(0), 1))
        x_dot_w_yi = x_dot_w_yi.squeeze()
        self.cos_t = x_dot_w_yi/(self.x_norm + self.eps)
        self.cos_t = torch.clamp(self.cos_t, -1.0, 1.0)

        self.k = torch.LongTensor(self.cos_t.size())
        for i in range(label.size(0)):
            self.k[i] = self.find_k(self.cos_t[i])

        if self.is_cuda:
            self.k = self.k.cuda()

        self.cos_mt = self.calc_cos_mt(self.cos_t)

        indictor = 1-2*(self.k%2)

        f_new = ((indictor.float()*self.cos_mt - 2*self.k.float())*self.x_norm).unsqueeze(1)

        for i in range(label.size(0)):
            out[i, label[i, 0]] = (self.beta*out[i, label[i, 0]]+ f_new[i, 0]) / \
                               (1+self.beta)

        return out

    def backward(self, grad_output):
        input, label = self.saved_tensors
        indictor = 1-2*(self.k%2)
        
        input_grad = torch.mm(grad_output, self.weight)
        
        # d_four_x_coeff: d_cos4x/dx = (32*cos3x^3-16cosx)* dcosx_dx
        d_four_x_coeff = (32*torch.pow(self.cos_t, 3) - 16*self.cos_t)

        w_choose = self.weight[label.view(-1)]
        dcos_dx = w_choose/(self.x_norm+self.eps).unsqueeze(1).expand_as(w_choose) - \
                  input*(self.cos_t/(self.x_norm*self.x_norm + self.eps)).unsqueeze(1).expand_as(input)

        
        coeff_w = indictor.float()*d_four_x_coeff
        coeff_x = ( indictor.float()*self.cos_mt-2*self.k.float()-indictor.float()*d_four_x_coeff*self.cos_t) / \
                (self.x_norm + self.eps )
        
        # norm the coeff_w
        total_norm = torch.sqrt(torch.pow(coeff_w, 2) + torch.pow(coeff_x, 2)) + self.eps
        coeff_w = coeff_w / total_norm
        coeff_x = coeff_x / total_norm

        df_dx = coeff_w.unsqueeze(1).expand_as(w_choose)*w_choose + \
                coeff_x.unsqueeze(1).expand_as(input)*input

        alpha = self.beta/(1+self.beta)
        grad_scale = torch.gather(grad_output, 1, label.view(label.size(0), 1))
        grad_scale = grad_scale.squeeze()
        input_grad += (1.0-alpha)*grad_scale.unsqueeze(1).expand_as(df_dx)*(df_dx-w_choose)
        
        # update weight using guli's method
        for i in range(input.size(0)):
            x_norm = input[i, :] /( self.x_norm[i] + self.eps )
            self.weight[label[i], :] += x_norm

        return input_grad, None, None
