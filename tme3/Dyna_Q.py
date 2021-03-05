
import numpy as np
from collections import defaultdict


class Dyna_Q(object):
    def __init__(self, action_space, explorer, gamma, alpha, alpha_R, alpha_P, k):
        self.action_space = action_space
        self.explorer = explorer
        self.Q = defaultdict(lambda : np.zeros(4))
        self.old_at = None
        self.old_st= None
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_P = alpha_P
        self.alpha_R = alpha_R
        self.k = k

        self.P = defaultdict(lambda : defaultdict(lambda : np.zeros(4)))
        self.R = defaultdict(lambda : defaultdict(lambda : np.zeros(4)))

        # P[st+1][st][a] = P(s_t+1 | st, a)
        # R[st][s_t+1][at] = R(st, at, s_t+1)


    # after execution of a_(t-1) we go to st and we get reward = r_(t-1)
    def act(self, s_t, reward, done):
        s_t = str(s_t.tolist())
        a_t = self.explorer.choose(s_t = s_t,Q = self.Q)
        self.learn(self.old_st, self.old_at, reward, s_t)
        self.old_at = a_t
        self.old_st = s_t
        return a_t

    def learn(self, old_st, old_at, reward, s_t):
        max = self.Q[s_t].max()
        self.Q[old_st][old_at] = self.Q[old_st][old_at] + self.alpha * (reward + self.gamma * max  - self.Q[old_st][old_at] )

        # update the model
        self.R[old_st][s_t][old_at] = self.R[old_st][s_t][old_at] + self.alpha_R * (reward - self.R[old_st][s_t][old_at])

        self.P[s_t][old_st][old_at] =  self.P[s_t][old_st][old_at] + self.alpha_R * (int((s_t != old_st)) - self.P[s_t][old_st][old_at])

        states = np.random.choice(np.array(list(self.P.keys())) , size = self.k )
        actions = np.array( [self.action_space.sample() for i in range(self.k) ] )
        for state, action in zip(states, actions):
            d1 = self.Q[state][action]
            d2 = sum(self.P[FutureState][state][action] * (self.R[state][FutureState][action] + self.gamma * self.Q[FutureState].max()) for FutureState in self.P) - d1
            self.Q[state][action] =  d1+ self.alpha *(d2)
