# -*- coding: utf-8 -*-

import torch.nn as nn
from qanet.layer_norm import LayerNorm1d

# The class conv_block is equivalent to the conv_block method in the original source code
class ConvolutionLayer(nn.Module):
    def __init__(self, num_conv_layers, kernel_size, num_filters,
               is_training = True, dropout = 0.0, d=128):
        
        super(ConvolutionLayer, self).__init__()
        self.layer_norm = LayerNorm1d()
        
        # Depthwise convolution. The less groups, the bigger saving on the number of parameter
        self.depthwise_separable_convolution = nn.Conv2d(d,d,1,1,0,groups=2)
        self.relu = nn.ReLU()
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        
    def forward(self, inputs, sublayers):
        outputs = inputs.unsqueeze(2)
        l,L = sublayers
        for i in range(self.num_conv_layers):
            if i%2 == 0:
                initialized_dropout = nn.Dropout(p=(1-self.dropout))
                outputs = initialized_dropout(outputs)
            # equivalent of norm_fn at line 121 of QANet.layers.py
            #outputs = self.layer_norm(outputs)
            outputs = self.depthwise_separable_convolution(outputs)
            outputs = self.relu(outputs)

        return outputs.squeeze(2), l