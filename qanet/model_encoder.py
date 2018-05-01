#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from qanet.encoder_block import EncoderBlock

class ModelEncoder(nn.Module):
    
    def __init__(self, num_blocks=7, num_conv=2, kernel_size=7, num_filters=512):
        super(ModelEncoder, self).__init__()
        
        self.num_blocks = num_blocks
        
        self.stackedEncoderBlocks = nn.ModuleList([EncoderBlock(num_conv=num_conv,
                                                                kernel_size=kernel_size,
                                                                num_filters=num_filters) for i in range(num_blocks)])
        
        
    def forward(self, x):
        
        # x is the following concatenation : [c, a, c*a, c*b], where a and b are
        # respectively a row of attention matrix A and B, c is a row of the 
        # embedded context matrix
        
        # first stacked model encoder blocks
        for i in range(self.num_blocks):
            x = self.stackedEncoderBlocks[i](x)
        M0 = x
        
        # second stacked model encoder blocks
        for i in range(self.num_blocks):
            x = self.stackedEncoderBlocks[i](x)
        M1 = x
        
        # third stacked model encoder blocks
        for i in range(self.num_blocks):
            x = self.stackedEncoderBlocks[i](x)
        M2 = x
        
        return M0, M1, M2
        
