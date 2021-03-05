import pickle
import argparse
import sys
import matplotlib

import gym
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from random import random
from pathlib import Path
import numpy as np
import os

from discriminator import *
from data_utils import *


gamma = 0.98
delta = 0.8
lmbda = 0.95
K = 10
eps = 0.2
ent_weight = 1e-3
eta = 1e-2
layers_discriminator = [256]
hidden_layer_v = 256
hidden_layer_pi = 256
lrs=[3e-4, 3e-4, 3e-4]
eps_clip = 0.1

class GAIL(nn.Module):
    """
    Gail agent : Generative Adversarial Imitation Learning
    """
    def __init__(self, env, opt, device, path_to_expert_data):
        super(GAIL, self).__init__()
        self.epoch, self.iteration = 0, 0
        self.feature_extractor = opt.featExtractor(env)
        
        # device (cuda if available else cpu)
        self.device = device
        
        # dimension of state space and number of possible actions
        self.state_dim = env.observation_space.shape[0]
        self.nb_actions = env.action_space.n
        
        # the discriminator
        self.discriminator = Discriminator(device, self.state_dim, self.nb_actions, layers_discriminator).double()
        
        
        # ppo kl networks
        self.v = nn.Sequential(
            nn.Linear(self.state_dim, hidden_layer_v),
            nn.ReLU(),
            nn.Linear(hidden_layer_v, 1)).double()
        self.pi = nn.Sequential(
            nn.Linear(self.state_dim, hidden_layer_pi),
            nn.ReLU(),
            nn.Linear(hidden_layer_pi, self.nb_actions),
            nn.Softmax(dim=-1),
        ).double()
        
        # optimizers, one for each network
        self.optimizer_disc = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=lrs[0]
        )
        self.optimizer_v = torch.optim.Adam(params=self.v.parameters(), lr=lrs[1])
        self.optimizer_pi = torch.optim.Adam(params=self.pi.parameters(), lr=lrs[2]) 
        
        # expert data
        self.exert_dataset = ExpertDataset(env, path_to_expert_data, device)
        self.expert_states, self.expert_actions = self.exert_dataset.get_expert_data()
        
        # create a memory 
        self.memory = []
        
        # push self to device
        self.to(device)
        
    def get_action(self, observation, rewrd, done):
        """
            Action with respect to the current policy (ppo networks : pi)
            Returns action and corresponding probability
        """
        with torch.no_grad():
            # get feature representation to pass to the network
            featues = self.feature_extractor.getFeatures(observation)
            # distribution over possible actions
            prob_on_states = self.pi(torch.tensor(featues, dtype=float).to(device))
            # get action that maximises log likelihood
            action = prob_on_states.argmax(dim=-1).item()
        return action, prob_on_states[action]
    
    def add_to_memory(self, ob, action, prob_action, reward, new_ob , done):
        """
        Stores a new transition in the memory
        Adds a new transition to the memory (ob, new_ob, action, reward, done(boolean), probabaility of action)
        """
        self.memory.append(
            [
                self.feature_extractor.getFeatures(ob).reshape(-1),
                self.feature_extractor.getFeatures(new_ob).reshape(-1),
                action,
                reward,
                float(done),
                float(prob_action),
            ]
        )
        
    def sample_expert(self, n_samples):
        """
        Get n samples ((state, action) couples) from the expert's trajectory (with shuffle)
        Returns (states, actions(one hot format))
        """
        # shuffle the data (n_sample may be > self.expert_states.size(0))
        ids = torch.randint(self.expert_states.size(0), (n_samples,))
        # sample transitions
        states = self.expert_states[ids]
        actions = self.expert_actions[ids]
        return states, actions

    def sample_agent(self):
        """
        Sample shuffled (state, actions) couples of memory
        Returns (states, actions(one hot format))
        """
        
        # get all states and actions from the memory
        states, actions = (
            torch.tensor([transition[0] for transition in self.memory], device=device, dtype=float),
            torch.tensor([transition[2] for transition in self.memory], device=device, dtype=float),
        )
        # shuffle the data
        ids = torch.randperm(states.size(0))
        # get shuffeld transitions from memory (iidize)
        states = states[ids]
        actions = actions[ids]
        actions = self.exert_dataset.toOneHot(actions)
        return states, actions

    def load_memory(self):
        """
        get all elements of memory
        Returns ([ob], [new_ob], [action](action id), [reward], [done], [prob_actions])
        """
        obs, new_obs, actions, rewards, done, prob_actions = (
            torch.tensor([transition[0] for transition in self.memory], device=device, dtype=float),
            torch.tensor([transition[1] for transition in self.memory], device=device, dtype=float),
            torch.tensor([transition[2] for transition in self.memory], device=device, dtype=int),
            torch.tensor([transition[3] for transition in self.memory], device=device, dtype=float).view(
                -1, 1
            ),
            torch.tensor([transition[4] for transition in self.memory], device=device, dtype=float).view(
                -1, 1
            ),
            torch.tensor([transition[5] for transition in self.memory], device=device, dtype=float),
        )

        return obs, new_obs, actions, rewards, done, prob_actions

    def train(self, beta):
        for _ in range(1):
            
            ####### DISCRIMINATOR : LOSS + GRADIENT STEP #######

            mem_size = len(self.memory)
            # get samples from expert and agent (all available)
            x_expert, x_agent = self.sample_expert(mem_size), self.sample_agent()

            # the discriminator's loss (as defined in literature)
            
            # add noise 
            states_agent = x_agent[0] + (torch.randn(x_agent[0].size(), device = self.device) * eta)
            actions_agent = x_agent[1] + (torch.randn(x_agent[1].size(), device = self.device) * eta)
            
            states_expert = x_expert[0] + (torch.randn(x_expert[0].size(), device = self.device) * eta)
            actions_expert = x_expert[1] + (torch.randn(x_expert[1].size(), device = self.device) * eta)
            
            disc_on_expert = self.discriminator(states_expert, actions_expert)
            disc_on_agent = self.discriminator(states_agent, actions_agent)
            disc_loss = torch.log(disc_on_expert) + torch.log(1 - disc_on_agent)

            # gradient step for the discriminator
            self.optimizer_disc.zero_grad()
            disc_loss = -disc_loss.mean()
            disc_loss.backward()
            self.optimizer_disc.step()

            # add noise to the discriminator's parameters (N(0, eta))
            """with torch.no_grad():
                for param in self.discriminator.parameters():
                    param.add_(torch.randn(param.size(), device = self.device) * eta)"""

            ####### END OF DISCRIMINATOR UPDATE #######
            
            
        
        for _ in range(K):
            
            ####### PPO V : LOSS + GRADIENT STEP #######

            # load memory (action's done wrt to current policy)
            state, new_state, action, reward, done, prob_sample = self.load_memory()

            # critic (ppo : v network)
            v = self.v(state).flatten()

            # v's loss log(discriminator(s_t, a_t))
            r_t = torch.log(self.discriminator(state, self.exert_dataset.toOneHot(action)).flatten())
            # clip rewards \in [-100, 0]
            r_t = torch.where(r_t > -100, r_t, torch.Tensor([-100.]).double().to(self.device))
            R = [r_t[i:].mean() for i in range(mem_size)]
            adv = torch.stack(R).to(device=self.device, dtype=float)
            v_loss = F.smooth_l1_loss(v, adv.detach()) 

            # gradient step for the v network
            self.optimizer_v.zero_grad()
            v_loss.backward()
            self.optimizer_v.step()

            ####### END OF PPO V UPDATE #######
            
            
            ####### PPO PI : LOSS + GRADIENT STEP #######
            
            # get actor's policy
            pi = self.pi(state)
            # actor's probability for actions in memory
            prob = self.pi(state).gather(1, action.unsqueeze(-1))
            
            ratio = torch.exp(
                torch.log(prob) - torch.log(prob_sample)
            )  # a/b = exp(log(a)-log(b))
            surr1 = ratio * adv.detach()
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv.detach()
            L = -torch.min(surr1, surr2)
            L = L.mean()
            H = (pi * torch.log(pi)).mean(dim=-1)
            H = H.mean()
            # one gradient step for pi
            self.optimizer_pi.zero_grad()
            (L - ent_weight * H).backward()
            self.optimizer_pi.step()
            
            ####### END OF PPO PI UPDATE #######            
            self.epoch += 1
            
            
        
            
        # clear memory
        self.memory = []
        # one grafient iteration
        self.iteration += 1
        
        with torch.no_grad():
            prob_new_actions = self.pi(state).gather(1, action.unsqueeze(-1))
            d_kl = (
                prob_new_actions
                * (torch.log(prob_new_actions) / torch.log(prob_sample))
            ).mean()  # KL div btw theta k et theta k+1

        # return losses (v, pi, discriminator) and ppo metrics
        return disc_loss.item(),  L.item(), H.item(), v_loss.item(), d_kl.item(), disc_on_agent.mean(dim=0).item(), disc_on_expert.mean(dim=0).item()
    
        