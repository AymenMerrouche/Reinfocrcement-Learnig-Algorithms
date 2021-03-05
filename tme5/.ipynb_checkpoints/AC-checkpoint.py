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
from explorer import *
from memory import *
from utils import *


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils import *

#Hyperparameters
alpha = 0.0002
gamma         = 0.98
trajectory_size     = 20

class AgentActorCritic(nn.Module):
    def __init__(self,env, opt):
        super(AgentActorCritic, self).__init__()
        self.buffer = [] # pour stoker les transitions

        self.opt, self.env = opt, env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        #==================================================================================
        #                            parametres
        #==================================================================================
        self.linear = nn.Linear(self.featureExtractor.outSize,256)
        self.linear_pi = nn.Linear(256,self.action_space.n)
        self.linear_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def R_t(self, r_tensor):
        """ fonction qui calcul les R_t ou rewards discounté et normalisé"""
        rewards = []
        dis_reward = 0
        for reward in torch.flip(r_tensor,(0,1)): # onflit pour commencer de la fin
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward) # oninsere au debut

        # normalisation
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        return rewards

    def train(self):

        s, a, r, s_prime, done = self.assembler_transitions()
        #==================================================================================
        #                            version Grid world
        #==================================================================================
        #v_phisprime = torch.cat([ self.V_phi(ex) for ex in s_prime ])
        #v_phis = torch.cat([ self.V_phi(ex) for ex in s ])

        #td_target = r + gamma * v_phisprime * done
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
        td_target = r + gamma * self.V_phi(s_prime) * done
        delta = td_target - self.V_phi(s)


        #pi = torch.cat([ self.PI_theta(ex, softmax_dim=1) for ex in s ])

        pi = self.PI_theta(s, softmax_dim=1) # ongenere les probas

        pi_a = pi.gather(1,a) # prendre ceux de l'action a
        # combinaison des deux loss, celle de policy gradient et la loss sur V_phi
        loss = -torch.log(pi_a) *delta.detach() + F.smooth_l1_loss(self.V_phi(s),td_target) # rewards.unsqueeze(-1) pour MC rollouty

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def PI_theta(self, x, softmax_dim = 0):
        """
        fonction qui calcul P_theta a partir d'un etat
        """
        x =  self.featureExtractor.getFeatures(x).float()
        x = F.relu(self.linear(x))
        x = self.linear_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
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

    def stock_transition(self, transition):
        """
        stockage d'une transition
        """
        self.buffer.append(transition)

    def assembler_transitions(self):

        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.buffer:
            s,a,r,s_prime,done, info = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            # on verifie avec le info.get('TimeLimit.truncated', False) si on s'est arreté car on est nul ou car on a bien fait
            done_mask = 0.0 if (done and not info.get('TimeLimit.truncated', False) )  else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)



        # vider le buffer
        self.buffer = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch





def main():
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    Trajectory_size = config["Trajectory_size"]

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + f"/AC_MC/"+ tstart


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

    agent = AgentActorCritic(env, config)
    print_interval = 10
    score = 0.0

    for i in range(episode_count):
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
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
            for t in range(trajectory_size):
                # on tire de la politique l'action
                prob = agent.PI_theta(torch.from_numpy(s).float())

                m = Categorical(prob)
                # on tire selon la distribution
                a = m.sample().item()
                s_prime, r, done, info = env.step(a) #
                # on stocke les transitions
                agent.stock_transition((s,a,r,s_prime,done, info))

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
