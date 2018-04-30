#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from qanet.residual_block import ResidualBlock, EncoderBlock

class ModelEncoder(nn.Module):
    
    def __init__(self, num_blocks=7, num_conv=2):
        super(ModelEncoder, self).__init__()
        
        self.num_blocks = num_blocks
        
        self.stackedEncoderBlocks = nn.ModuleList([EncoderBlock(num_conv=num_conv) for i in range(num_blocks)])
        
        
    def forward(self, x):
        
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
        
