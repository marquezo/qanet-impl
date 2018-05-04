# -*- coding: utf-8 -*-

from torch import nn
from qanet.encoder_block import EncoderBlock

class EmbeddingEncoder(nn.Module):
    
    def __init__(self, resize_in=500, hidden_size=128, resize_kernel=7, resize_pad=3,
                 n_blocks=1, n_conv=4, kernel_size=7, padding=3, 
                 conv_type='depthwise_separable', with_self_attn='True', n_heads=8, batch_size=32):
        super(EmbeddingEncoder, self).__init__()
        
        self.n_blocks = n_blocks
        
# =============================================================================
#         self.residual_block = ResidualBlock(num_blocks, num_conv_layers, kernel_size, mask=None,
#                  num_filters=128, input_projection=False, num_heads=8,
#                  seq_len=None, is_training=True, bias=True, dropout=0.0)
# =============================================================================
        self.resizeConvolution = nn.Conv1d(in_channels=resize_in,
                                            out_channels=hidden_size,
                                            kernel_size=resize_kernel,
                                            padding=resize_pad)
        
        self.stackedEncoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                n_filters=hidden_size,
                                                                conv_type=conv_type,
                                                                with_self_attn=with_self_attn,
                                                                n_heads=n_heads,
                                                                batch_size=batch_size) for i in range(n_blocks)])
    
    def forward(self, context_emb, question_emb):
        
        # resizing convolution        
        context_emb = self.resizeConvolution(context_emb)
        question_emb = self.resizeConvolution(question_emb)
        
        # weight sharing between context and question embedding encoders
        for i in range(self.n_blocks):
            context_emb = self.stackedEncoderBlocks[i](context_emb)
            question_emb = self.stackedEncoderBlocks[i](question_emb)

        return context_emb, question_emb