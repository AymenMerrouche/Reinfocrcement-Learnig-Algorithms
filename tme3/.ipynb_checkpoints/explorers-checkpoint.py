import numpy as np

from collections import defaultdict
class Explorer(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def choose(self):
        pass

class RandomExplorer(Explorer):
    def __init__(self, action_space):
        super().__init__(action_space)
    def choose(self, Q=None, s_t = None):
        return self.action_space.sample()

class Greedy(Explorer):
    def __init__(self, action_space):
        super().__init__(action_space)
    def choose(self,  Q=None, s_t = None):
        return np.argmax(Q[s_t])

class Epsilon_Greedy(Explorer):
    def __init__(self, action_space, epsilon):
        super().__init__(action_space)
        self.epsilon = epsilon
    def choose(self,  Q=None, s_t = None):
        if np.random.rand(1) < self.epsilon :
            return self.action_space.sample()
        return np.argmax(Q[s_t])

class Boltzman(Explorer):
    def __init__(self, action_space, T = 1e-3):
        super().__init__(action_space)
        self.T = T
    def choose(self,  Q, s_t):
        somme = sum(np.exp(Q[s_t])/self.T)
        p = (np.exp(Q[s_t])/self.T)/somme
        return np.argmax(p)

class UCB(Explorer):

    def __init__(self, action_space):
        super().__init__(action_space)
        self.visits = defaultdict(lambda : np.ones(4))
        pass
    def choose(self,  Q=None, s_t = None):
        a= np.argmax(Q[s_t] + np.sqrt(2 * np.log(self.visits[s_t].sum()) /self.visits[s_t] ))
        self.visits[s_t][a] += 1
        return a
