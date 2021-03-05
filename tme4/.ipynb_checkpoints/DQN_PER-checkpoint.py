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
class DQNAgent_PER(object):
    """ DQN Agent"""

    def __init__(self, env, opt, explorer, buffer_size, gamma, alpha, b,update_frequency, test = False):
        self.opt, self.env = opt, env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.explorer = explorer

        self.update_frequency = update_frequency
        # Q_theta function to approximate
        # TODO: add hidden sizes as parameter of the init
        self.Q_theta = NN(self.featureExtractor.outSize, self.action_space.n, [200]).cuda()
        # buffer size
        self.buffer_size = buffer_size
        self.epsilon = 0.4
        self.weight_decay = 0.99999
        self.buffer = Memory(self.buffer_size, prior = True)

        self.old_st, self.old_at = None, None

        self.gamma, self.alpha = gamma, alpha
        self.test = test
        # optimzer
        # TODO: specify the optimizer directly in the yaml files and use it here with opt
        self.optimizer = optim.Adam(self.Q_theta.parameters(), lr=self.alpha)
        self.criterion = torch.nn.SmoothL1Loss()# comme demandé

        self.episode_counter = 0

        self.q_target = None

        self.sample_size, self.stored_transitions = b, 0
    def act(self, s_t, reward, done):
        # update the target network, initilized as same
        ## TODO: make the 100 a paramater of the class
        self.updateQTarget(self.update_frequency)
        #
        s_t_featured = self.featureExtractor.getFeatures(s_t)
        #s_t_featured = np.array(s_t_featured.tolist())
        # mise à jour du nombre d'actions effectuées
        self.episode_counter +=1

        Q_theta_phi_st = self.Q_theta(torch.tensor(s_t_featured).float().cuda()).cpu().detach().numpy()
        if np.random.rand() <= self.epsilon :
            action = self.action_space.sample()
        else:
            action =  np.argmax(Q_theta_phi_st)
        # remplissage du buffer
        #pdb.set_trace()
        if self.old_st is not None: # if s_t is not none
            self.buffer.store((self.old_st, self.old_at, s_t_featured, reward, done))
            self.stored_transitions += 1

        # if in
        if  self.stored_transitions > 0 and not self.test : # not in a testing phase
            self.optimizer.zero_grad()
            # sampling random mini_batches from buffer
            idx, weights, samples = self.buffer.sample(min(self.stored_transitions, self.sample_size))
            # actions
            # a_j in the pseudo code
            a_j = np.array([sample[1] for sample in samples])
            # rewards in the form batch_size * 1
            # r_j in the pseudo code
            r_j = torch.tensor([sample[3] for sample in samples]).float()

            #  phi_j new states
            phi_j = torch.tensor([sample[0].reshape(-1) for sample in samples]).float()
            phi_jplus1 = torch.tensor([sample[2].reshape(-1) for sample in samples]).float()
            # mask sample that terminate at next step mask = 0 if done == true
            mask = torch.tensor([int( not sample[4]) for sample in samples])
            #calculate loss on sample
            with torch.no_grad():
                # if done is true (mask = 0), only take reward into acccount
                max = self.q_target(phi_jplus1.float().cuda()).cpu().max(dim=1).values

                y_j =  r_j + mask * self.gamma* max
            
            max_weight = np.max(weights)
            # the arange is here to specify the lines, the a_j the columns
            y_hat = self.Q_theta(phi_j.float().cuda())[ np.arange(len(a_j)), a_j ].flatten()
            loss = self.criterion(y_hat, y_j.cuda()) * max_weight
            self.buffer.update(idx, np.abs(y_hat.cpu().detach().numpy() - y_j.cpu().detach().numpy()))
            loss.backward()
            for p in self.Q_theta.parameters():
                p.grad *= max_weight
            self.optimizer.step()
            

        self.old_at = action if not done else None
        self.old_st = s_t_featured if not done else None
        if self.epsilon > 0.01:
            self.epsilon = self.epsilon * self.weight_decay
        return action
    def updateQTarget(self, C):
        """ function to update the Q learning target

            Attributes:
            C : update every n actions
        """
        if self.episode_counter % C == 0:
            self.q_target = copy.deepcopy(self.Q_theta)
    # TODO:  i'm not using it, but you can add it to save the models
    # there is a paramater in the yaml, accesed by opt.formfile
    # load it if it's true
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
