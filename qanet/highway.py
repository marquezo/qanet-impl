from torch import nn
import torch.nn.functional as F

class Highway(nn.Module):
    """ Version 1 : carry gate is (1 - transform gate)"""
    
    def __init__(self, input_size=500, n_layers=2):
        super(Highway, self).__init__()
        
        self.n_layers = n_layers
        
        self.transform = nn.ModuleList([nn.Linear(in_features=input_size, out_features=input_size) for i in range(n_layers)])
        self.fc = nn.ModuleList([nn.Linear(in_features=input_size, out_features=input_size) for i in range(n_layers)])
        
    def forward(self, x):
        
        for i in range(self.n_layers):
            transformed = F.sigmoid(self.transform[i](x))
            x = transformed * self.fc[i](x) + (1-transformed) * x
            x = F.relu(x)

        return x

class Highway_v2(nn.Module):
    """ Version 2 : carry gate is independent from transform gate """
    
    def __init__(self, input_size=500, n_layers=2):
        super(Highway_v2, self).__init__()
        
        self.n_layers = n_layers
        
        self.transform = nn.ModuleList([nn.Linear(in_features=input_size, out_features=input_size) for i in range(n_layers)])
        self.carry = nn.ModuleList([nn.Linear(in_features=input_size, out_features=input_size) for i in range(n_layers)])
        self.fc = nn.ModuleList([nn.Linear(in_features=input_size, out_features=input_size) for i in range(n_layers)])
        
    def forward(self, x):

        for i in range(self.n_layers):
            transformed = F.sigmoid(self.transform[i](x))
            carried = F.sigmoid(self.carry[i](x))
            x = transformed * self.fc[i](x) + carried * x
            x = F.relu(x)

        return x