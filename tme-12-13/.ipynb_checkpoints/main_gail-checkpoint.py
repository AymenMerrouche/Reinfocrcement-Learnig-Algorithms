import pickle
import argparse
import sys
import matplotlib

import gym
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from random import random
from pathlib import Path
import numpy as np
import os
from gail import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "./expert_data/"
file = "expert.pkl"
path_to_expert_data  = path+file
n_steps_to_gradient_step = 1000 # steps to take to update (v, pi d) (min size of memory (in case of episodes that take longer))
beta = 1e-3
gamma = 0.98
delta = 0.8


# load the environment
config = load_yaml("./configs/config_random_lunar.yaml")

env = gym.make(config.env)
if hasattr(env, "setPlan"):
    env.setPlan(config.map, config.rewards)
tstart = str(time.time()).replace(".", "_")
env.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
episode_count = config.nbEpisodes
ob = env.reset()

# define the agent
agent_id = f"_g{gamma}_delta{delta}_beta{beta}"
agent_dir = f'models/{config["env"]}/'
os.makedirs(agent_dir, exist_ok=True)
savepath = Path(f"{agent_dir}{agent_id}.pch")
agent = GAIL(env, config, device, path_to_expert_data)

# ---yaml and tensorboard---#
outdir = "./runs/" + config.env + "/GAIL/" + agent_id + "_" + tstart
print("Saving in " + outdir)
os.makedirs(outdir, exist_ok=True)
save_src(os.path.abspath(outdir))
write_yaml(os.path.join(outdir, "info.yaml"), config)
writer = SummaryWriter(outdir)
rsum = 0
mean = 0
verbose = True
itest = 0
reward = 0
done = False


logging_i = 0
steps_counter_grad_step = 0
n_steps_to_gradient_step = 64

for i in range(episode_count):
    if i % int(config.freqVerbose) == 0 and i >= config.freqVerbose:
        verbose = True
    else:
        verbose = False

    if i % config.freqTest == 0 and i >= config.freqTest:
        print("Test time! ")
        mean = 0
        agent.test = True

    if i % config.freqTest == config.nbTest and i > config.freqTest:
        print("End of test, mean reward=", mean / config.nbTest)
        itest += 1
        writer.add_scalar("rewardTest", mean / config.nbTest, itest)
        agent.test = False

    if i % config.freqSave == 0:
        with open(savepath, "wb") as f:
            torch.save(agent, f)
    j = 0
    if verbose:
        pass
        #env.render()
    verbose = False

    done = False
    
    
    while not (done):
        
        steps_counter_grad_step += 1
        if verbose:
            env.render()
        action, prob = agent.get_action(
            ob, reward, done
        )  # choose action and determine the prob of that action
        ob_new, reward, done, _ = env.step(action)  # process action
        agent.add_to_memory(
            ob, action, prob, reward, ob_new, done
        )  # storing the transition          

        ob = ob_new
        j += 1
        rsum += reward
        if done:
            print(f"{i} rsum={rsum}")
            writer.add_scalar("reward", rsum, i)
            agent.nbEvents = 0
            mean += rsum
            rsum = 0
            ob = env.reset()
    
        if (steps_counter_grad_step>= n_steps_to_gradient_step) and done:
            print("############## GRADIENT UPDATE ##############")
             # n_steps_to_gradient_step ore more depending on the last episode
        
            logging_i += 1

            # take a gradient step
            disc_loss,  L, H, v_loss, d_kl, disc_on_agent, disc_on_expert = agent.train(beta)
            
            # log mean rewards

            writer.add_scalar("adversial", disc_loss, logging_i)
            writer.add_scalar("ppo loss", v_loss, logging_i)
            
            writer.add_scalar("disc_on_agent", disc_on_agent, logging_i)
            writer.add_scalar("disc_on_expert", disc_on_expert, logging_i)
            
            steps_counter_grad_step = 0
        
env.close()