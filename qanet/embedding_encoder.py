# -*- coding: utf-8 -*-

from torch import nn
from qanet.encoder_block import EncoderBlock

class EmbeddingEncoder(nn.Module):
    
    def __init__(self, num_blocks=1, num_conv=4, kernel_size=7, mask=None,
                 resize_in=500, resize_out=128, resize_kernel=1, resize_stride=1, resize_pad=0,
                 num_filters=128, input_projection=False, num_heads=8,
                 seq_len=None, is_training=True, bias=True, dropout=0.0):
        super(EmbeddingEncoder, self).__init__()
        
        self.num_blocks = num_blocks
        
# =============================================================================
#         self.residual_block = ResidualBlock(num_blocks, num_conv_layers, kernel_size, mask=None,
#                  num_filters=128, input_projection=False, num_heads=8,
#                  seq_len=None, is_training=True, bias=True, dropout=0.0)
# =============================================================================
        self.resizeConvolution = nn.Conv1d(in_channels=resize_in,
                                            out_channels=resize_out,
                                            kernel_size=resize_kernel,
                                            stride=resize_stride,
                                            padding=resize_pad)
        
        self.encoderBlock = EncoderBlock(num_conv=num_conv,
                                         kernel_size=kernel_size,
                                         num_filters=num_filters,
                                         num_heads=num_heads)
    
    def forward(self, context_emb, question_emb):
        
        # resizing convolution        
        context_emb = self.resizeConvolution(context_emb)
        question_emb = self.resizeConvolution(question_emb)
        
        # weight sharing between context and question embedding encoders
        for i in range(self.num_blocks):
            context_emb = self.encoderBlock(context_emb)
            question_emb = self.encoderBlock(question_emb)

        return context_emb, question_emb