#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# taken from https://github.com/minsangkim142/QANet/blob/6540d9ad68f8a713b58f772d0ea5d4a8ff8eb27b/layers.py#L91-L109 - line 314

import torch
import math
from torch import nn
from torch.autograd import Variable

class PositionEncoding(nn.Module):
    
    def __init__(self):
        super(PositionEncoding, self).__init__()
    
    def forward(self, x, min_timescale=1.0, max_timescale=1.0e4):
        
        length = x.size()[1]
        channels = x.size()[2]
        
        # computing signal

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
        
        # change this
#        if torch.cuda.is_available():
#            signal = signal.cuda()

        return x + signal

    """
    Adds a bunch of sinusoids of different frequencies to a Tensor.
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

    """
    Gets a bunch of sinusoids of different frequencies.
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