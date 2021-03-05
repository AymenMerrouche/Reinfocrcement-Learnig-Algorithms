import torch
from torch import nn


class RandomAgent():
    """ A simple agent with NN """
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 3))

    def act(self, obs, reward, done, info):
        pass

    def play(self, obs, configuration):
        with torch.no_grad():
            return self.model(torch.tensor(obs).view(1,1).float()).argmax().item()
