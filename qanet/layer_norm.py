# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class LayerNorm1d(nn.Module):

    def __init__(self, n_features=128, eps=1e-6, affine=True):
        super(LayerNorm1d, self).__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)

        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y