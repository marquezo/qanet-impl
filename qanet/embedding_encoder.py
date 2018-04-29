# -*- coding: utf-8 -*-

from torch import nn
from qanet.residual_block import ResidualBlock

class EmbeddingEncoder(nn.Module):
    
    def __init__(self, num_blocks, num_conv_layers, kernel_size, mask=None,
                 num_filters=128, input_projection=False, num_heads=8,
                 seq_len=None, is_training=True, bias=True, dropout=0.0):
        super(EmbeddingEncoder, self).__init__()
        self.residual_block = ResidualBlock(num_blocks, num_conv_layers, kernel_size, mask=None,
                 num_filters=128, input_projection=False, num_heads=8,
                 seq_len=None, is_training=True, bias=True, dropout=0.0)
    
    def forward(self, context, question):
        
        context_result = self.residual_block(context)
        question_result = self.residual_block(question)

        return context_result, question_result