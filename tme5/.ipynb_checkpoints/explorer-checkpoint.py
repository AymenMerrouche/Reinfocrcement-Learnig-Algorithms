from collections import defaultdict
import matplotlib
import pdb;
#matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt


"""
def decay(epsilon_0,t,T,epsilon_f):
    eta = (epsilon_0 - epsilon_f) / (epsilon_f*T)
"""
class Explorer(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def choose(self):
        pass

class RandomExplorer(Explorer):
    def __init__(self, action_space):
        super().__init__(action_space)
    def choose(self, estimations, s_t = None):
        return self.action_space.sample()

class Greedy(Explorer):
    def __init__(self, action_space):
        super().__init__(action_space)
    def choose(self,  estimations, s_t = None):
        return np.argmax(estimations)

class Epsilon_Greedy(Explorer):
    def __init__(self, action_space, epsilon):
        super().__init__(action_space)
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.001
    def choose(self,  estimations, s_t, episode):
        if self.epsilon > self.epsilon_min: # Epsilon Update
            self.epsilon = ((5000-episode)/5000) 
        if np.random.rand() <= self.epsilon :
            return self.action_space.sample()
        return np.argmax(estimations)
    
class Boltzman(Explorer):
    def __init__(self, action_space, T = 0.000000001):
        super().__init__(action_space)
        self.T = T
    def choose(self, estimations, s_t = None):
        somme = sum(np.exp(estimations)/self.T)
        p = (np.exp(estimations)/self.T)/somme
        #return np.argmax(np.random.multinomial(1, p))
        return np.argmax(p)

class UCB(Explorer):

    def __init__(self, action_space):
        super().__init__(action_space)
        self.visits = defaultdict(lambda : np.ones(action_space.n))
        pass
    def choose(self,  estimations, s_t = None):
        a= np.argmax(estimations + np.sqrt(2 * np.log(self.visits[str(s_t.tolist())].sum()) /self.visits[str(s_t.tolist())] ))
        self.visits[str(s_t.tolist())][a] += 1
        return a
