#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from qanet.encoder_block import EncoderBlock

class ModelEncoder(nn.Module):
    
    def __init__(self, n_blocks=7, n_conv=2, kernel_size=7, padding=3,
                 hidden_size=128, conv_type='depthwise_separable', 
                 with_self_attn='True', n_heads=8, batch_size=8):
        super(ModelEncoder, self).__init__()
        
        self.n_blocks = n_blocks
        
        self.stackedEncoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                n_filters=hidden_size,
                                                                conv_type=conv_type,
                                                                with_self_attn=with_self_attn,
                                                                n_heads=n_heads,
                                                                batch_size=batch_size) for i in range(n_blocks)])
        
    def forward(self, x):
        
        # x is the following concatenation : [c, a, c*a, c*b], where a and b are
        # respectively a row of attention matrix A and B, c is a row of the 
        # embedded context matrix
        
        # first stacked model encoder blocks
        for i in range(self.n_blocks):
            x = self.stackedEncoderBlocks[i](x)
        M0 = x
        
        # second stacked model encoder blocks
        for i in range(self.n_blocks):
            x = self.stackedEncoderBlocks[i](x)
        M1 = x
        
        # third stacked model encoder blocks
        for i in range(self.n_blocks):
            x = self.stackedEncoderBlocks[i](x)
        M2 = x
        
        return M0, M1, M2
        
