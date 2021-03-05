import argparse
import sys
import matplotlib
from pathlib import Path
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from memory import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils import *



class AgentActorCritic(nn.Module):
    def __init__(self,env, opt, alpha, gamma ):
        super(AgentActorCritic, self).__init__()
        # params alpha lr, gamma discount
        self.alpha, self.gamma = alpha, gamma

        self.opt, self.env = opt, env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        #==================================================================================
        #                            parametres
        #==================================================================================
        # utilisation d'un linear patagé entre pi et v (ca marche mieux de cette maniere)
        self.linear = nn.Linear(self.featureExtractor.outSize,128)
        # deuxieme couche pour pi
        self.linear_pi = nn.Linear(128,self.action_space.n)
        # deuxieme couche pour v
        self.linear_v = nn.Linear(128,1)
        # un seul optimizeur pour les deux ( de ce que je peux voir ca se fait couremement)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # nouveau buffer customisé pas Memory car on a pas de max ici
        self.buffer = Buffer()

    def R_t(self, r_tensor):
        """ fonction qui calcul les R_t ou rewards discounté et normalisé"""
        rewards = []
        dis_reward = 0
        for reward in torch.flip(r_tensor,(0,1)): # on flip pour commencer de la fin
            dis_reward = reward + self.gamma * dis_reward
            rewards.insert(0, dis_reward) # on insere au debut

        # normalisation
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        return rewards

    def train(self):

        s, a, r, s_prime, done = self.buffer.sample()
        #==================================================================================
        #                            version Grid world
        #==================================================================================
        #v_phisprime = torch.cat([ self.V_phi(ex) for ex in s_prime ])
        #v_phis = torch.cat([ self.V_phi(ex) for ex in s ])

        #td_target = r + self.gamma * v_phisprime * done
        #delta = td_target - v_phis


        #==================================================================================
        #                            version montecarlo R_t
        #==================================================================================
        #rewards = self.R_t( r) # for montecarlo rollout
        # version pour lunar et cart
        #advantage = rewards -  self.V_phi(s)
        #==================================================================================
        #                            version TD(0)
        #==================================================================================
        td_target = r + self.gamma * self.V_phi(s_prime) * done
        delta = td_target - self.V_phi(s)


        #pi = torch.cat([ self.PI_theta(ex) for ex in s ])

        pi = self.PI_theta(s, True) # on genere les probas

        pi_a = pi.gather(1,a) # prendre ceux de l'action a
        # combinaison des deux loss, celle de policy gradient et la loss sur V_phi
        loss = -torch.log(pi_a) *delta.detach() + F.smooth_l1_loss(self.V_phi(s),td_target) # rewards.unsqueeze(-1) pour MC rollouty
        #==================================================================================
        #                           optimizer
        #==================================================================================
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def PI_theta(self, x, batch = False):
        """
        fonction qui calcul P_theta a partir d'un etat
        """
        x =  self.featureExtractor.getFeatures(x).float()
        x = F.relu(self.linear(x))
        x = self.linear_pi(x)
        if batch :
            prob = F.softmax(x, dim=1)
        else :
            prob = F.softmax(x, dim=0)
        return prob

    def V_phi(self, x):
        """
        fonction qui calcul V_phi a partir d'un etat
        """

        x = self.featureExtractor.getFeatures(x)

        x = torch.tensor(x).float()

        x = F.relu(self.linear(x))

        v = self.linear_v(x)

        return v






def main():
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    trajectory_size = config["trajectory_size"] # max

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + f"/AC/"+ tstart


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

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    Test = False
    # ================
    # declaration de l'Agent
    # ================

    agent = AgentActorCritic(env, config, config["alpha"], config["gamma"])

    for i in range(episode_count):
        if i % freqTest == 0 and i >= freqTest:
            print("Test time! ")
            mean = 0
            Test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            Test = False

        done = False
        s = env.reset()
        while not done:
            for j in range(trajectory_size):
                # on tire de la politique l'action
                prob = agent.PI_theta(torch.from_numpy(s).float())

                m = Categorical(prob)
                # on tire selon la distribution
                a = m.sample().item()
                s_prime, r, done, info = env.step(a) #
                # on stocke la transition
                agent.buffer.store((s,a,r,s_prime,done, info))

                s = s_prime
                rsum += r

                if done:
                    print(str(i) + " rsum=" + str(rsum) )
                    logger.direct_write("reward", rsum, i)
                    agent.nbEvents = 0
                    agent.episode = i
                    mean += rsum
                    rsum = 0

                    break;
            agent.train()


    env.close()

if __name__ == '__main__':
    main()
