import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from torch.optim import Adam
import numpy as np
from utils import *
import math


class Behavioral_Cloning(nn.Module):
    """
    Behavioral Cloning class. Modelisation of the pi_theta function.
    """
    def __init__(self, input_size, output_size, layers = [64, 32], activation = nn.Tanh()):
        """
        :param input_size: dimension of the states space.
        :param output_size: number of possible actions.
        :param layers: specify the layers of the network (by default [64, 32]).
        :parama activation: nonlinearity to use (by defaul tanh).
        """        
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.activation = activation
        
        # just unroll the ffd's layers
        if len(layers)>0:
            fc = [nn.Linear(input_size, self.layers[0], bias = True)]
            fc.append(activation)
            for i in range(len(self.layers)-1):
                fc.append(nn.Linear(self.layers[i], self.layers[i+1], bias = True))
                fc.append(activation)
            fc.append(nn.Linear(self.layers[-1], output_size, bias = True))
        else:
            fc = [nn.Linear(input_size, output_size, bias = True)]
        self.pi_theta = nn.Sequential(*fc)
        
    def forward(self, x):
        # compute network's output
        return self.pi_theta(x)