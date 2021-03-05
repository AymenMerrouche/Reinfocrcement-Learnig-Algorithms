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
        self.policy, self.value,_ = valueIteration(statedic, mdp, epsilon, gamma)
    def act(self, observation, reward, done):
        return self.policy[str(observation.tolist())]

class PolicyIterationAgent(object):
    def __init__(self, statedic, mdp, epsilon = 0.01, gamma= 0.95):
        self.policy, self.value, _ = policyIteration(statedic, mdp, epsilon, gamma)
    def act(self, observation, reward, done):
        return self.policy[str(observation.tolist())]


def test_agent(plan, episode_count,mode, alpha, gamma,distri = {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1}):


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # for each type of cases we associate a reword
    env.setPlan("gridworldPlans/plan" +plan+  ".txt", distri)

    # statedict : key : string of state "[[], []...] " : int number of the state # nombre de clé >
    # mdp { string of state : {int de l'action : [(proba, futureState, reward, done)] } } # nombre de clé
    statedic, mdp = env.getMDP()  # recupere le mdp , statedic
    # policy : { state : action}
    if mode == "PolicyIteration":
        agent = PolicyIterationAgent(statedic, mdp,  alpha, gamma)
    else:
        agent = ValueIterationAgent(statedic, mdp,  alpha, gamma )
    fig = drawValuePolicyMap(agent)
    #plt.savefig(f"./graphs/plan{sys.argv[1]}_epsilon0.01_gamma0.95.png")
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = episode_count
    reward = 0
    done = False
    rewards = []
    iterations = []
    FPS = 0.0001

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = False#(i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        iterations.append(j)
        rewards.append(rsum)
    print("done")
    env.close()
    return np.arange(1, episode_count+1),np.array(rewards),np.array(iterations), fig



def nb_states():
    states = []
    for plan in [0,1,2,3,4,5,6,7,8,10]:
        env = gym.make("gridworld-v0")
        env.seed(0)  # Initialise le seed du pseudo-random
        # Faire un fichier de log sur plusieurs scenarios
        outdir = 'gridworld-v0/random-agent-results'
        envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
        # for each type of cases we associate a reword
        env.setPlan("gridworldPlans/plan" +str(plan)+  ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

        # statedict : key : string of state "[[], []...] " : int number of the state # nombre de clé >
        # mdp { string of state : {int de l'action : [(proba, futureState, reward, done)] } } # nombre de clé
        statedic, mdp = env.getMDP()  # recupere le mdp , statedic
        # policy : { state : action}
        states.append(len(statedic))
        env.close()

    return np.array([0,1,2,3,4,5,6,7,8,10]), np.array(states)


if __name__=="__main__":
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan11.txt", {0: -0.05, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    #print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    #print(state)  # un etat du mdp
    #print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = ValueIterationAgent(statedic, mdp,  0.001, 0.99 )


    episode_count = 2
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 1 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
