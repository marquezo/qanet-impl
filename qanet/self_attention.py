# -*- coding: utf-8 -*-

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Multi-head attention mechanism as defined in Vaswani et al. 2017
class SelfAttention(nn.Module):
    def __init__(self, batch_size, n_heads, n_filters):
        super().__init__()
        self.batch_size = batch_size
        self.dimension_contexts = n_filters // n_heads
        self.dimension_values = n_filters // n_heads
        self.n_filters = n_filters
        self.n_heads = n_heads
        

        
        self.weights_queries = [torch.zeros(batch_size, self.dimension_contexts, n_filters).cuda() for _ in range(n_heads)]
        self.weights_contexts = [torch.zeros(batch_size, self.dimension_contexts, n_filters).cuda() for _ in range(n_heads)]
        self.weights_values = [torch.zeros(batch_size, self.dimension_values, n_filters).cuda() for _ in range(n_heads)]
        self.weights_output = torch.zeros(batch_size, self.dimension_values * n_heads, n_filters).cuda()
        
        nn.init.xavier_normal(self.weights_output)
        for i in range(n_heads):
            nn.init.xavier_normal(self.weights_queries[i])
            nn.init.xavier_normal(self.weights_contexts[i])
            nn.init.xavier_normal(self.weights_values[i])
            
            
    def forward(self, inputs):
        assert inputs.size()[1] == self.n_filters
        array_weights_queries = []
        array_weights_contexts = []
        array_weights_values = []

        for i in range(self.n_heads):
            array_weights_queries.append(torch.bmm(self.weights_queries[i], inputs.data))
            array_weights_contexts.append(torch.bmm(self.weights_contexts[i], inputs.data))
            array_weights_values.append(torch.bmm(self.weights_values[i], inputs.data))
        heads = []
        for i in range(self.n_heads):
            outputs = torch.bmm(array_weights_queries[i].transpose(1, 2), array_weights_contexts[i])
            outputs = torch.div(outputs, math.sqrt(self.dimension_contexts))
            outputs = F.softmax(outputs, dim=1)
            headi = torch.bmm(array_weights_values[i], outputs.data)
            heads.append(headi)
        head = torch.cat(heads, dim=1)
        outputs = torch.bmm(self.weights_output, head)
        return outputs          
        
        