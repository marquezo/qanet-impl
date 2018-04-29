from torch import nn
import torch.nn.functional as F

class Highway(nn.Module):
    """ Version 1 : carry gate is (1 - transform gate)"""
    
    def __init__(self, input_size):
        super(Highway_v1, self).__init__()
        
        self.transform1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.fc1 = nn.Linear(in_features=input_size, out_features=input_size)
        
        self.transform2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.fc2 = nn.Linear(in_features=input_size, out_features=input_size)
        
    def forward(self, x):

        transformed = F.sigmoid(self.transform1(x))
        x = transformed * self.fc1(x) + (1-transformed) * x
        
        #activation fct?
        x = F.relu(x)
        
        transformed = F.sigmoid(self.transform2(x))
        x = transformed * self.fc2(x) + transformed * x
        
        return x

class Highway_v2(nn.Module):
    """ Version 2 : carry gate is independent from transform gate """
    
    def __init__(self, input_size):
        super(Highway_v2, self).__init__()
        
        self.transform1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.carry1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.fc1 = nn.Linear(in_features=input_size, out_features=input_size)
        
        self.transform2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.carry2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.fc2 = nn.Linear(in_features=input_size, out_features=input_size)
        
    def forward(self, x):

        transformed = F.sigmoid(self.transform1(x))
        carried = F.sigmoid(self.carry1(x))
        x = transformed * self.fc1(x) + carried * x
        
        #activation fct?
        x = F.relu(x)
        
        transformed = F.sigmoid(self.transform2(x))
        carried = F.sigmoid(self.carry2(x))
        x = transformed * self.fc2(x) + carried * x
        
        x = F.relu(x)
        
        return x