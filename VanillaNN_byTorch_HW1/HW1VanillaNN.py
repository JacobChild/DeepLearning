# HW1VanillaNN.py
# Jacob Child
# Jan 15, 2026

import torch 
from torch import nn

class VanillaNN(nn.Module):
    
    def __init__(self, ninf, noutf):
        super().__init__() # super is a reference to the parent class, in this case nn.Module, so it is being initialized
        
        self.network = nn.Sequential(
            # These are hyper parameters, number of layers and number of nodes in each layer
            nn.Linear(ninf, 32), # nn.Linear(input, output), this is the connection from the input layer to the first hidden layer
            nn.ReLU(),
            nn.Linear(32, 32), # hidden layer 1 to hidden layer 2
            nn.ReLU(),
            nn.Linear(32, noutf) #! make sure to end on linear layer, this is the connection from the last hidden layer to the output layer
        )
        
    def forward(self, x):
        return self.network(x)