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
import torch.optim as optim


# TODO: add a dataclass to handle the pramameters going into the class, it's a bit messy
class DQNGAgent(object):
    """ DQN Agent"""

    def __init__(self, env, opt, explorer, device, buffer_size, gamma, alpha, b,update_frequency, test = False):
        
        # config parameters
        self.opt, self.env = opt, env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.explorer = explorer
        self.device = device

        self.update_frequency = update_frequency
        # Q_theta function to approximate
        self.Q_theta = NN(self.featureExtractor.outSize * 2, self.action_space.n, [200]).to(self.device)
        # buffer size
        self.buffer_size = buffer_size
        # epsilon greedy parameter
        self.epsilon = 0.2
        # define the buffer
        self.buffer = Memory(self.buffer_size, prior = False)
        # discount actor and learning rate
        self.gamma, self.alpha = gamma, alpha
        self.test = test
        # optimzer
        self.optimizer = optim.Adam(self.Q_theta.parameters(), lr=self.alpha)
        self.criterion = torch.nn.SmoothL1Loss()# comme demandé

        self.episode_counter = 0

        self.q_target = None

        self.sample_size, self.stored_transitions = b, 0
        
    def act(self, s_t, goal, reward, done):
        self.episode_counter +=1
        
        # update the target network, initilized as same
        self.updateQTarget(self.update_frequency)
        # get s_t featured, goal is already featured in main
        s_t_featured = s_t.reshape(-1)
        goal = goal.reshape(-1)
        
        # target network takes state representation concatenated with goal representation
        input_target = torch.tensor(np.concatenate((s_t_featured, goal))).to(self.device)
        
        # action to take (epsilon greedy)
        Q_theta_phi_st = self.Q_theta(input_target.float()).detach().cpu().numpy()
        if np.random.rand() <= self.epsilon :
            action = self.action_space.sample()
        else:
            action =  np.argmax(Q_theta_phi_st)

        # our buffer is not empty and if we're not testing
        if  ((self.stored_transitions > 0) and (self.episode_counter % 10 == 0)) and not self.test : # not in a testing phase
            
            self.optimizer.zero_grad()
            
            # sampling random mini_batches from buffer
            samples = self.buffer.sample(min(self.stored_transitions, self.sample_size))
            # actions
            # a_j in the pseudo code
            a_j = np.array([sample[1] for sample in samples])
            # rewards in the form batch_size * 1
            # r_j in the pseudo code
            r_j = torch.tensor([sample[4] for sample in samples]).float().to(self.device)
            
            
            #  phi_j new states
            phi_j = torch.tensor([np.concatenate((sample[0].reshape(-1),sample[2].reshape(-1)), 0) for sample in samples]).float().to(self.device)
            phi_jplus1 = torch.tensor([np.concatenate((sample[3].reshape(-1),sample[2].reshape(-1)), 0) for sample in samples]).float().to(self.device)
            # mask sample that terminate at next step mask = 0 if done == true
            mask = torch.tensor([int( not sample[5]) for sample in samples]).to(self.device)
            #calculate loss on sample
            with torch.no_grad():
                # if done is true (mask = 0), only take reward into acccount
                max = self.Q_theta(phi_jplus1.float()).max(dim=1).values
                y_j =  r_j + mask * self.gamma* max
            # the arange is here to specify the lines, the a_j the columns
            loss = self.criterion(self.Q_theta(phi_j.float())[ np.arange(len(a_j)), a_j ].flatten(), y_j)
            loss.backward()
            self.optimizer.step()
        return action
    def updateQTarget(self, C):
        """ function to update the Q learning target

            Attributes:
            C : update every n actions
        """
        if self.episode_counter % C == 0 :
            self.q_target = copy.deepcopy(self.Q_theta)
    def save(self,outputDir):
        """ function to save the Q function"""
        with Path(outputDir).open("wb") as fp:
            torch.save(self.Q_theta, fp)
        pass

    def load(self,inputDir):
        """ function to save the Q function"""
        if Path(inputDir).is_file():
            with savepath.open("rb") as fp:
                self.Q_theta = torch.load(fp)
        pass
