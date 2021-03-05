import torch
import torch.nn as nn
from torch.distributions import Categorical




class Discriminator(nn.Module):
    """
    GAIL's Discriminator neural network
    """
    def __init__(self, device, state_dim, nb_actions, layers = [64, 32]):
        super(Discriminator, self).__init__()
        
        # just create a fully connected feedforward
        self.input_layer = nn.Linear(state_dim+nb_actions, layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.output_layer = nn.Linear(layers[-1], 1)
        
        # number of possible actions
        self.nb_actions = nb_actions
        
        # cpu \cuda
        self.device = device
        

    def feed_ffn(self, state, action):
        """Compute the feedforward network's output"""
        # create a one hot encoder for action
        action_tensor_onehot = action.double()
        # conctenate with state
        state_action = torch.cat([state, action_tensor_onehot], dim=1).to(self.device)
        
        # just unroll the fc feddfwd
        x = self.input_layer(state_action)
        x = torch.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        # project on 1 (prob of discriminator)
        x = self.output_layer(x)
        return x

    def forward(self, state, action):
        """return logit for being generated (S, A)"""
        # compute nn's result and then sigmoid
        logit = self.feed_ffn(state, action)
        prob = torch.sigmoid(logit)
        
        return prob