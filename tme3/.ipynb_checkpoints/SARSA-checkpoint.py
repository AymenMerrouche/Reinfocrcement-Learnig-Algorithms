import numpy as np
from collections import defaultdict

class SARSA(object):
    def __init__(self, action_space, explorer, gamma, alpha):
        self.action_space = action_space
        self.explorer = explorer
        self.Q = defaultdict(lambda : np.zeros(4))
        self.old_at = None
        self.old_st= None
        self.gamma = gamma
        self.alpha = alpha
    # after execution of a_(t-1) we go to st and we get reward = r_(t-1)
    def act(self, s_t, reward, done):
        s_t = str(s_t.tolist())
        a_t = self.explorer.choose(s_t = s_t,Q = self.Q)
        self.learn(self.old_st, self.old_at, reward, s_t)
        self.old_at = a_t
        self.old_st = s_t
        return a_t

    def learn(self, old_st, old_at, reward, s_t):
        action_t1 = self.explorer.choose(s_t = s_t,Q = self.Q)
        self.Q[old_st][old_at] = self.Q[old_st][old_at] + self.alpha * (reward + self.gamma *  self.Q[s_t][action_t1]  - self.Q[old_st][old_at] )
