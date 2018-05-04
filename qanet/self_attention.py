# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import use_cuda

# Multi-head attention mechanism as defined in Vaswani et al. 2017
class SelfAttention(nn.Module):
    def __init__(self, batch_size=8, n_heads=8, n_filters=128):
        super().__init__()
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.key_dim = n_filters // n_heads
        self.value_dim = n_filters // n_heads
        
        self.fc_query = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_key = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_value = nn.ModuleList([nn.Linear(n_filters, self.value_dim) for i in range(n_heads)])
        self.fc_out = nn.Linear(n_heads * self.value_dim, n_filters)
            
    def forward(self, x):
        
        l = x.shape[1]
        
        heads = Variable(torch.zeros(self.n_heads, self.batch_size, l, self.value_dim))
        if use_cuda:
            heads = heads.cuda()

        for i in range(self.n_heads):
            Q = self.fc_query[i](x)
            K = self.fc_key[i](x)
            V = self.fc_value[i](x)
            
            # scaled dot-product attention
            tmp = torch.bmm(Q.permute(0, 2, 1), K)
            tmp = tmp / np.sqrt(self.key_dim)
            tmp = F.softmax(tmp, dim=1)
            
            heads[i] = torch.bmm(V, tmp)

        # concatenation is the same as reshaping our tensor
        x = heads.view(self.batch_size, l, -1)
        x = self.fc_out(x)
        return x      
        
if __name__ == "__main__":

    batch_size = 8
    l = 60
    n_filters = 128
    
    mdl = SelfAttention()
    
    x = torch.ones(batch_size, l, n_filters)
    x = Variable(x)
    
    print(mdl(x))