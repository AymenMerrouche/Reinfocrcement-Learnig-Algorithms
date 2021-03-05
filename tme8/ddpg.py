import argparse
import sys
import matplotlib
from pathlib import Path
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from torch.utils.tensorboard import SummaryWriter

import gym
import gridworld
import random
import collections
import numpy as np

from memory import *


class DDPG:
    """
    classe implementant DDPG
    """
    def __init__(self, epsilon_mu, epsilon_q, gamma, ro, batch_size):

        self.epsilon_mu, self.epsilon_q = epsilon_mu, epsilon_q

        self.gamma, self.ro = gamma, ro
        self.batch_size = batch_size
        # buffer customisé qui renvoi un tensor
        self.buffer = Buffer()
        # declaration des reseaux Q et MU
        self.q, self.q_target = Q(), Q()

        self.mu, self.mu_cible = mu(), mu()
        #initailisation des poinds de q_target et mu_cible
        self.q_target.load_state_dict(self.q.state_dict())

        self.mu_cible.load_state_dict(self.mu.state_dict())
        # optimiseur de mu
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.epsilon_mu)
        # optimiseur de q
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=self.epsilon_q)
        #  Orn_Uhlen noise
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        pass

    def train(self):
        """
        train procedure pour ddpg
        """
        # on tire depuis le buffer
        s,a,r,s_prime,done_mask  = self.buffer.sample(self.batch_size)
        # td 0
        target = r + self.gamma * self.q_target(s_prime, self.mu_cible(s_prime)) * done_mask
        # optimise en td 0
        q_cost = F.smooth_l1_loss(self.q(s,a), target.detach())

        self.q_optimizer.zero_grad()
        q_cost.backward()
        self.q_optimizer.step()
        # loss de Mu
        mu_cost = - self.q(s,self.mu(s)).mean()

        self.mu_optimizer.zero_grad()
        mu_cost.backward()
        self.mu_optimizer.step()


    def update(self,origin, target):
        """
        mise a jour des target pour un net et un
        """
        for param_target, param in zip(target.parameters(), origin.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.ro) + param.data * self.ro)


if __name__ == '__main__':
    #config = load_yaml('./configs/config_pendulum.yaml')
    #config = load_yaml('./configs/config_MountainCarContinuous.yaml')
    config = load_yaml('./configs/config_random_lunar.yaml')
    freqTest = config["freqTest"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + f"/ddpg/"+ tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))

    agent = DDPG(config["epsilon_mu"], config["epsilon_q"], config["gamma"], config["ro"], config["batch_size"])

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    Test = False
    Learn = True
    score= 0


    for i in range(3000):

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            Test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            Test = False
            Test = False

        s = env.reset()
        done = False

        while not done:
            a = agent.mu(torch.from_numpy(s).float())
            a = a.detach().numpy() + agent.ou_noise()[:1]
            s_prime, r, done, info = env.step(a)
            agent.buffer.store((s,a,r/100.0,s_prime,done, info))
            score +=r
            s = s_prime
            rsum += r
        # si on depasse 6000 transiitions
        if agent.buffer.size()>6000:
            # j'ai mis K a 20 ici direct
            for j in range(20):
                agent.train()
                agent.update(agent.mu, agent.mu_cible)
                agent.update(agent.q,  agent.q_target)

        logger.direct_write("reward", rsum, i)
        print(str(i) + " rsum=" + str(rsum) )

        mean += rsum
        rsum = 0
        score = 0.0

    env.close()
