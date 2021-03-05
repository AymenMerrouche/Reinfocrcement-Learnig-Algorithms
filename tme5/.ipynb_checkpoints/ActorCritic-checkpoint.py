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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# TODO: add a dataclass to handle the pramameters going into the class, it's a bit messy
class BaseActorCritic(nn.Module):
    def __init__(self,env, opt,  gamma, alpha,update_frequency):
        super(BaseActorCritic, self).__init__()
        self.opt, self.env = opt, env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.gamma = gamma
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.iteration_counter = 0
        # Q_theta function to approximate
        # TODO: add hidden sizes as parameter of the init
        self.buffer = []

        self.pi = NN(self.featureExtractor.outSize, self.action_space.n, [30,30])
        self.v = NN(self.featureExtractor.outSize, 1, [30,30])
        self.optim_pi = optim.Adam(self.pi.parameters(), lr=alpha)
        self.optim_v = optim.Adam(self.v.parameters(), lr=alpha)

    def PI_theta(self, x, softmax_dim = 0):
        x = self.featureExtractor.getFeatures(x)
        x = x.float()
        x = self.pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    def V_phi(self, x):
        x = self.featureExtractor.getFeatures(x)
        x = x.float()
        v = self.v(x)
        return v
    def V_phi_target(self, x):
        with torch.no_grad():
            x = self.featureExtractor.getFeatures(x)
            x = x.float()
            x = F.relu(self.fc1(x))
            v = self.fc_v_target(x) # sans softmax
        return v
    def store_transition(self, transition):
        self.buffer.append(transition)
        pass
    def assemble_samples(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.buffer:
            s,a,r,s_prime,done, info = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if (done and not info.get('TimeLimit.truncated', False) )else 1.0
            done_lst.append([done_mask])

            s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                       torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                       torch.tensor(done_lst, dtype=torch.float)
        self.buffer = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train(self):
        s, a, r, s_prime, done = self.assemble_samples()

        td_target = r + self.gamma * self.V_phi(s_prime) * done
        delta = td_target - self.V_phi(s)

        pi = self.PI_theta(s, softmax_dim=1)
        pi_a = pi.gather(1,a)

        loss = -torch.log(pi_a) * delta.detach()
        self.optim_pi.zero_grad()
        loss.mean().backward()
        self.optim_pi.step()

        loss = F.smooth_l1_loss(self.V_phi(s), td_target.detach())
        self.optim_v.zero_grad()
        loss.backward()
        self.optim_v.step()


    def updateVTarget(self):
        """ function to update the V_phi taget
            Attributes:
            C : update every n actions
        """
        if self.iteration_counter % self.update_frequency == 0:
            self.fc_v_target = copy.deepcopy(self.fc_v)

if __name__ == '__main__':
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
    outdir = "./XP/" + config["env"] + f"/AC/"+ tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()
    #=======================================================================================
    ### explorer
    agent = BaseActorCritic(env, config,gamma=config["gamma"], alpha = config["alpha"],update_frequency = config["update_frequency"])
    #==============================================================================================
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
        agent.buffer = [] # empty buffer
        while not done:
            for t in range(Trajectory_size):
                prob = agent.PI_theta(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                agent.store_transition((s,a,r,s_prime,done, info))

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

            if not Test:
                agent.train()


    env.close()
