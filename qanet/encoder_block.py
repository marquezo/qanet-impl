#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
import torch.nn.functional as F

from qanet.position_encoding import PositionEncoding
from qanet.layer_norm import LayerNorm1d
from qanet.self_attention import SelfAttention
from qanet.depthwise_separable_conv import DepthwiseSeparableConv1d


class EncoderBlock(nn.Module):
    
    def __init__(self, n_conv, kernel_size=7, padding=3, n_filters=128, n_heads=8,
                 conv_type='depthwise_separable', batch_size=32):
        super(EncoderBlock, self).__init__()

        self.n_conv = n_conv        
        self.n_filters = n_filters
        
        self.positionEncoding = PositionEncoding(n_filters=n_filters)
        self.layerNorm = LayerNorm1d(n_features=n_filters)
        # sticking to normal convolutions for now
        if conv_type == 'normal':
            self.conv = nn.ModuleList([nn.Conv1d(n_filters,
                                                 kernel_size=kernel_size,
                                                 padding=padding) for i in range(n_conv)])
        elif conv_type == 'depthwise_separable':
            self.conv = nn.ModuleList([DepthwiseSeparableConv1d(n_filters=n_filters,
                                                                kernel_size=kernel_size,
                                                                padding=padding) for i in range(n_conv)])
        self.selfAttention = SelfAttention(batch_size, n_heads, n_filters)
        self.fc = nn.Linear(in_features=n_filters, out_features=n_filters)
        
    def forward(self, x):
        
        x = self.positionEncoding(x)
        
        # convolutional layers
        for i in range(self.n_conv):
            tmp = self.layerNorm(x)
            tmp = F.relu(self.conv[i](tmp))
            x = tmp + x
    
        # self attention
        # tmp = self.layerNorm(x)
        # tmp = self.selfAttention(x)
        # x = tmp + x
        
        # fully connected layer
        tmp = self.layerNorm(x)
        
        #which dimension is fully conencted?
        tmp = tmp.permute(0, 2, 1)
        tmp = self.fc(tmp)
        tmp = tmp.permute(0, 2, 1)
        x = x + tmp
        
        return x