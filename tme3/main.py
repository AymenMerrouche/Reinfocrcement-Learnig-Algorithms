
from collections import defaultdict
import matplotlib
import pdb;
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from policy_finder import valueIteration, policyIteration
import copy
import sys
from utils import *
import matplotlib.pyplot as plt
from Dyna_Q import *
from SARSA import *
from Q_learning import *
from explorers import *
import time

if __name__ == '__main__':
    config = load_yaml('./configs/test.yaml')
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # for each type of cases we associate a reword
    env.setPlan("gridworldPlans/plan"+ str(config["map"])+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    action_space = env.action_space
    explorerRandom = RandomExplorer(action_space)
    explorerEpsilonGreedy = Epsilon_Greedy(action_space, epsilon = config["epsilon"])
    explorerUCB = UCB(action_space)
    explorerBoltzman = Boltzman(action_space)
    explorerGreedy = Greedy(action_space)


    if config["explorer"] == "Epsilon_Greedy":
        explorer = explorerEpsilonGreedy
    elif config["explorer"] == "Boltzman":
        explorer = explorerBoltzman
    elif config["explorer"] == "UCB":
        explorer = explorerUCB
    elif config["explorer"] == "Greedy":
        explorer = explorerGreedy

    else:
        explorer = explorerRandom
    if config["alg"] == "Q_learning":
        agent = QLearning(action_space,  explorer,gamma = config["gamma"], alpha = config["alpha"])
    elif config["alg"] == "SARSA":
        agent = SARSA(action_space,  explorer,gamma = config["gamma"], alpha = config["alpha"])
    elif config["alg"] == "Dyna_Q":
        agent = Dyna_Q(action_space, explorer, gamma = config["gamma"], alpha= config["alpha"], alpha_R= config["alpha_R"], alpha_P= config["alpha_P"], k= config["k"])

    #fig = drawValuePolicyMap(agent)
    #plt.savefig(f"./graphs/plan{sys.argv[1]}_epsilon0.01_gamma0.95.png")
    rewards, actions = [], []
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = config["episode_count"]
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    t0 = time.time()
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = False#( i % config["frequence"]== 0 and i > 0)  # afficher 1 episode sur 100
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
                rewards.append(rsum)
                actions.append(j)
                break

    print("done")

    env.close()
    print("===============================")
    save = "./tests/" + config["alg"] + "_" +config["explorer"] + "_" + str(config["map"])
    episodes = np.arange(1, episode_count +1)
    np.savetxt(save + "_rewards.csv", np.array(rewards))
    np.savetxt(save + "_actions.csv", np.array(actions))

    print("         END")
    t1 = time.time()
    print(" time : ", t1-t0)
