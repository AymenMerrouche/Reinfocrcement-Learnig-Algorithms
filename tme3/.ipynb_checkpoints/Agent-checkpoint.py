import matplotlib
import pdb;
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from policy_finder import valueIteration, policyIteration
import copy
import sys
from utils import drawValuePolicyMap
import matplotlib.pyplot as plt


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class ValueIterationAgent(object):
    """agent that folows value ietration policy"""
    def __init__(self, statedic, mdp, epsilon = 0.01, gamma= 0.95):
        self.policy, self.value = valueIteration(statedic, mdp, epsilon, gamma)
    def act(self, observation):
        return self.policy[str(observation.tolist())]

class PolicyIterationAgent(object):
    def __init__(self, statedic, mdp, epsilon = 0.01, gamma= 0.95):
        self.policy, self.value = policyIteration(statedic, mdp, epsilon, gamma)
    def act(self, observation):
        return self.policy[str(observation.tolist())]
if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # for each type of cases we associate a reword
    env.setPlan("gridworldPlans/plan" +sys.argv[1]+  ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    # statedict : key : string of state "[[], []...] " : int number of the state # nombre de clé >
    # mdp { string of state : {int de l'action : [(proba, futureState, reward, done)] } } # nombre de clé
    statedic, mdp = env.getMDP()  # recupere le mdp , statedic
    # policy : { state : action}

    agent = PolicyIterationAgent(statedic, mdp,  0.01, gamma = 0.95)
    #fig = drawValuePolicyMap(agent)
    #plt.savefig(f"./graphs/plan{sys.argv[1]}_epsilon0.01_gamma0.95.png")
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = int(sys.argv[2])
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    print("done")
    env.close()
