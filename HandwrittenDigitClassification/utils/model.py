import torch
from torch import nn
from utils.datasetloading import *

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size=1*28*28
output_size=10

class NeuralNetwork(nn.Module):
    

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers =nn.Sequential(
            nn.Linear(input_size,120),
            nn.ReLU(),
            nn.Linear(120,80),
            nn.ReLU(),
            nn.Linear(80,output_size),
            )
        
    def forward(self,x):
        result=self.flatten(x)
        logits=self.layers(result)
        return logits


