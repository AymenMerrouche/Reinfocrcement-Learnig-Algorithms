import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Q(nn.Module):

    def __init__(self, nbAgents, input_dim, action_dim):

        super(Q, self).__init__()

        self.nbAgents = nbAgents
        self.input_dim = input_dim
        self.action_dim = action_dim

        etats_dim = input_dim * nbAgents

        act_dim = self.action_dim * nbAgents
        # =============================================
        #                   parametres
        # =============================================

        self.lineaire1 = nn.Linear(etats_dim, 1024)
        self.lineaire2 = nn.Linear(1024+act_dim, 512)
        self.lineaire3 = nn.Linear(512, 300)
        self.lineaire4 = nn.Linear(300, 1)

    # etats: batch_size * etats_dim
    def forward(self, etats, acts):
        result = F.relu(self.lineaire1(etats))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.lineaire2(combined))
        return self.lineaire4(F.relu(self.lineaire3(result)))


class Mu(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Mu, self).__init__()
        self.lineaire1 = nn.Linear(input_dim, 500)
        self.lineaire2 = nn.Linear(500, 128)
        self.lineaire3 = nn.Linear(128, action_dim)

    def forward(self, etats):
        result = F.relu(self.lineaire1(etats))
        result = F.relu(self.lineaire2(result))
        result = F.tanh(self.lineaire3(result))
        return result
