# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from qanet.layer_norm import LayerNorm1d

# The class conv_block is equivalent to the conv_block method in the original source code
class ConvolutionLayer(nn.Module):
    def __init__(self, num_conv_layers, kernel_size, num_filters,
               is_training = True, dropout = 0.0, sublayers = (1, 1), d=128):
        
        super(ConvolutionLayer, self).__init__()
        self.layer_norm = LayerNorm1d()
        # QUESTION : for depthwise convolution, we need to specify the group.
        # in tensorflow we have tf.nn.separable_conv2d function. but I can't see where the group is defined
        # temporarily put the groups value to 2 to make it work in the meantime
        self.depthwise_separable_convolution = nn.Conv2d(d,d,1,1,0,groups=2)
        self.relu = nn.ReLU()
        self.sublayers = sublayers
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        
    def forward(self, inputs):
        outputs = inputs.unsqueeze(2)
        l,L = self.sublayers
        for i in range(self.num_conv_layers):
            residual = outputs
            if i%2 == 0:
                initialized_dropout = nn.Dropout(p=(1-self.dropout))
                outputs = initialized_dropout(outputs)
            # equivalent of norm_fn at line 121 of QANet.layers.py

            #outputs = self.layer_norm(outputs)
            outputs = self.depthwise_separable_convolution(outputs)
            outputs = self.relu(outputs)

        return outputs.squeeze(2), l