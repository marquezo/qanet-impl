# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from qanet.convolution_layer import ConvolutionLayer
from qanet.self_attention import SelfAttention

class ResidualBlock(nn.Module):
    
    def __init__(self, num_blocks, num_conv_layers, kernel_size, mask=None,
                 num_filters=128, input_projection=True, num_heads=8,
                 seq_len=None, is_training=True, bias=True, dropout=0.0):
        super(ResidualBlock, self).__init__()
        
        self.num_blocks = num_blocks
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.mask = mask
        self.num_filters = num_filters
        self.input_projection = input_projection
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.is_training = is_training
        self.bias = bias
        
        # Input size, output size, kernel size, stride, padding
        self.resize_convolution = nn.Conv1d(500,128,1,1,0)
        self.conv_block = ConvolutionLayer(num_conv_layers=num_conv_layers, 
                                           kernel_size=kernel_size, num_filters=num_filters, 
                                           dropout=dropout, is_training=is_training, bias=bias)
        self.self_attention = SelfAttention(num_filters=num_filters, seq_len=seq_len, 
                                              dropout=dropout, mask=mask, num_heads=num_heads,
                                              is_training=is_training, bias=bias)

        
    def forward(self, inputs):
        if self.input_projection:
            # with 1d convolution, we will resize input from 500 to 128
            # TODO : check how to initialize weights and bias
            inputs = self.resize_convolution(inputs)
            
        outputs = inputs
        sublayer = 1
        total_sublayers = (self.num_conv_layers + 2) * self.num_blocks
        for i in range(self.num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = self.conv_block(outputs, (sublayer, total_sublayers))
            outputs, sublayer = self.self_attention(outputs, (sublayer, total_sublayers))
        return outputs

# this method allow attention to learn to use absolute and relative positions
# cf https://github.com/minsangkim142/QANet/blob/6540d9ad68f8a713b58f772d0ea5d4a8ff8eb27b/layers.py#L91-L109 - line 314
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = torch.arange(0,length)
    position = position.type(torch.FloatTensor)
    num_timescales = channels // 2
    # print(channels)
    # print(num_timescales-1)
    num_timescales = float(num_timescales)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(0,num_timescales) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
#     tf.pad(input, [padding line, padding column]) : https://www.tensorflow.org/api_docs/python/tf/pad. by default, pad with 0
#     signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
# TODO : atm padding does nothing. check how to pad right column only, in pytorch ?
    padding = nn.ConstantPad2d(0,0)
    signal = padding(signal)
    signal = signal.view(1,length, channels)
    
    if torch.cuda.is_available:
        signal = signal.cuda()
    return signal

