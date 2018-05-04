#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# taken from https://github.com/minsangkim142/QANet/blob/6540d9ad68f8a713b58f772d0ea5d4a8ff8eb27b/layers.py#L91-L109 - line 314
# position encoding as proposed by Vaswani et al. 2017

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from constants import use_cuda

class PositionEncoding(nn.Module):
    
    def __init__(self, n_filters=128, min_timescale=1.0, max_timescale=1.0e4):
        super(PositionEncoding, self).__init__()
        
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.d = n_filters
        
        # we use the fact that cos(x) = sin(x + pi/2) to compute everything with one sin statement
        self.freqs = torch.Tensor(
                [max_timescale**(-i/self.d) if i%2==0 else max_timescale**(-(i-1)/self.d) for i in range(self.d)]).unsqueeze(1)
        self.phases = torch.Tensor([0 if i%2==0 else np.pi/2 for i in range(self.d)]).unsqueeze(1)

        if use_cuda:
            self.freqs = self.freqs.cuda()
            self.phases = self.phases.cuda()

    def forward(self, x):
        
        l = x.shape[-1]
        
        # computing signal
        pos = torch.arange(l).repeat(self.d, 1)
        if use_cuda:
            pos = pos.cuda() 
        tmp = pos * self.freqs + self.phases
        pos_enc = torch.sin(tmp)
        pos_enc = Variable(pos_enc)
        x = x + pos_enc

        return x
    
if __name__ == "__main__":
    
    mdl = PositionEncoding()
    
    batch_size=8
    n_channels=128
    n_items=60
    
    input = Variable(torch.ones(batch_size, n_channels, n_items))
    
    if use_cuda:
        input = input.cuda()
    
    out = mdl(input)