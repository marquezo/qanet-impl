#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
import torch.nn.functional as F

from qanet.position_encoding import PositionEncoding
from qanet.layer_norm import LayerNorm1d
from qanet.self_attention import SelfAttention


class EncoderBlock(nn.Module):
    
    def __init__(self, num_conv, kernel_size, num_filters=128, num_heads=8):
        super(EncoderBlock, self).__init__()

        self.num_conv = num_conv        
        self.num_filters = num_filters
        
        self.positionEncoding = PositionEncoding()
        self.layerNorm = LayerNorm1d(num_features=num_filters)
        self.conv = nn.ModuleList([nn.Conv1d(num_filters,num_filters,1,1,0,groups=2) for i in range(num_conv)])
        self.selfAttention = SelfAttention()
        self.fc = nn.Linear(in_features=num_filters, out_features=num_filters)
        
    def forward(self, x):
        
        x = self.positionEncoding(x)
        
        # convolutional layers
        for i in range(self.num_conv):
            
            tmp = self.layerNorm(x)
            tmp = F.relu(self.conv[i](tmp))
            x = tmp + x
    
        # self attention
        tmp = self.layerNorm(x)
        tmp = self.selfAttention(x)
        x = tmp + x
        
        # fully connected layer
        tmp = self.layerNorm(x)
        
        #which dimension is fully conencted?
        tmp = tmp.permute(0, 2, 1)
        tmp = self.fc(tmp)
        tmp = tmp.permute(0, 2, 1)
        x = x + tmp
        
        return x