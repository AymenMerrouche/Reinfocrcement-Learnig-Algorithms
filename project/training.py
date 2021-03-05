import pickle
import os
import kaggle_environments as kg
import pdb
from DQN_agents.DQN import *
# Authorized libraries : Python Standard Library, gym, numpy, scipy, pytorch (cpu only)

import torch
from DQN_agents.utils import *
from torch.utils.tensorboard import SummaryWriter
from DQN_agents.memory import *
from DQN_agents.utils import *
import torch.optim as optim
import pdb

import numpy as np
import torch
import sys

try:
    """ For kaggle environment """
    sys.path.append("/kaggle_simulations/agent")
except:
    pass

from agents.randomagent import RandomAgent


tstart = str(time.time())
tstart = tstart.replace(".", "_")
    
outdir = "./XP/DQN/"  + tstart

print("Saving in " + outdir)
os.makedirs(outdir, exist_ok=True)
save_src(os.path.abspath(outdir))


logger = LogMe(SummaryWriter(outdir))


env = kg.make("rps",debug=True,
        configuration = {
            "actTimeout" : 1,
            "agentTimeout": 60,
            "runTimeout" : 1200 })


agent1 = RandomAgent()

agent2 = DQNAgent(buffer_size = 10000, gamma=0.999, alpha = 0.01, b= 200,update_frequency= 100,  test = False )


info = env.reset()

rsum = 0
mean = 0
verbose = True
itest = 0
reward = 0
done = False




for i in range(150):
    action1 = np.random.randint(0,2)
    action2 = np.random.randint(0,2)
    obs_1, obs_2 = env.step([action1, action2])
    count_plays = 0
    while True:
        action1 = np.random.randint(0,2)
        
        state = np.array([obs_2.observation['remainingOverageTime'], obs_2.observation['reward'], obs_2.observation['lastOpponentAction']])
        ob, reward, done = state, obs_2.reward, obs_2.status == "DONE",
        action2 = agent2.act( ob , reward , done )
        obs_1, obs_2 = env.step([action1, int(action2)])
        count_plays += 1
        rsum += obs_2.reward
        if obs_2.status == "DONE":
            print(str(i) + " rsum=" + str(rsum), count_plays )
            logger.direct_write("reward", rsum/1000, i)
            agent2.nbEvents = 0
            agent2.episode = i
            mean += rsum
            rsum = 0
            ob = env.reset()
            break




with open("agents/DQNagent.pkl","wb") as f:
        pickle.dump(agent2, f)
