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
from DQN_GOAL import *
import numpy as np


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = load_yaml('./configs/config_random_gridworld.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    # 
    env.setPlan("gridworld/gridworldPlans&Goals/plan2Multi.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/DQNG/" + tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()
    #=======================================================================================
    ### explorer
    explorer = Epsilon_Greedy(env.action_space, 0.2 )
    #explorer = UCB(env.action_space)
    #explorer = Greedy(env.action_space)
    agent = DQNGAgent(env,config,  explorer, device ,buffer_size = int(1e6), gamma=0.99, alpha = 1e-3, b= 1000, update_frequency= 1000,  test = False )
    #==============================================================================================
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    writer = SummaryWriter(outdir)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    mem_temp = []
    
    effective_episodes = 1
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 :
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            writer.add_scalar("reward", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
            
        # sample the goal for the forthcoming episode
        goal, _ = env.sampleGoal()
        goal = agent.featureExtractor.getFeatures(goal)
        ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()
            # one action wrt to the current agent
            action = agent.act(ob, goal, reward, done)
            # new observation + featurize
            new_ob, _, _, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            # done if we attained the goal
            done = (new_ob == goal).all()
            # sparse rewards
            reward = 1.0 if done else -0.1
            
            
            # filling the buffer
            # [old_st, old_at, old_goal, new_st, new_goal, reward, done]
            agent.buffer.store((ob, action, goal, new_ob, reward, done))
            agent.stored_transitions += 1
            
            # next iteration
            ob = new_ob
            j += 1
            rsum += reward
            
            if done or j >= 100 :
                print("epsiode = " + str(i) + " | rsum=" + str(rsum) + ", " + str(j))
                mean += rsum
                rsum = 0
                ob = env.reset()
                break
    env.close()
