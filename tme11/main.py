
from MADDPG import MADDPG
import torch

import matplotlib

import argparse
import sys
import matplotlib
from pathlib import Path
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import pdb
from utils import *
from torch.utils.tensorboard import SummaryWriter

import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy

import traceback
import warnings
import sys

warnings.filterwarnings("ignore")



"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world


"""if __name__ == '__main__':


    env,scenario,world = make_env('simple_spread')

    o = env.reset()
    reward = []
    for _ in range(200):
        a = []
        for i, _ in enumerate(env.agents):
            a.append((np.random.rand(2)-0.5)*2)
        o, r, d, i = env.step(a)

        print("========================================================================")
        print(o, r, d, i)

        reward.append(r)
        env.render(mode="none")
    print(reward)


    env.close()
"""
if __name__ == '__main__':
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_yaml('./configs/config_simple_spread.yaml')
    env,scenario,world = make_env(config["env"])

    freqTest = config["freqTest"]
    nbTest = config["nbTest"]



    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + f"/maddpg/"+ tstart

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


    np.random.seed(1234)
    torch.manual_seed(1234)
    env.seed(1234)

    nbAgents = len(env.agents)
    nbEtats = config["nbEtats"]
    nbActions = config["nbActions"]
    capacity = config["capacity"]
    batch_size = config["batch_size"]

    n_episode = config["n_episode"]
    max_steps = config["max_steps"]
    eploration_phase = config["eploration_phase"]


    maddpg = MADDPG(nbAgents, nbEtats, nbActions, batch_size, capacity,
                    eploration_phase)

    for i_episode in range(n_episode):

        if i_episode % freqTest == 0 and i_episode >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            Test = True

        if i_episode % freqTest == nbTest and i_episode > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            Test = False

        obs = env.reset()
        # =============================================
        # padding pour les actions pas de la meme taille
        # (peut etre faux mais bon)
        # =============================================
        obs_zeros = np.zeros((len(obs), nbEtats))
        for index ,i in enumerate(obs):
            obs_zeros[index,:i.shape[0]] = i
        obs = obs_zeros

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        total_reward = 0.0
        for t in range(max_steps):
            obs = obs.float().to(device)
            # choix des actions
            action = maddpg.choose(obs).data.cpu()
            # execution
            obs_, reward, done, _ = env.step(action.numpy())
            # transformer en tensor
            reward = torch.tensor(reward).float().to(device)
            # padding pour les cas ou on a pas la meme dim pour tout les
            # agents (maybe c'est faux)
            obs_zeros = np.zeros((len(obs_), nbEtats))
            for index ,i in enumerate(obs_):
                obs_zeros[index,:i.shape[0]] = i
            obs_ = obs_zeros
            obs_ = torch.from_numpy(obs_).float()
            # si on atteint le nombre de steps
            # le prochain etat est nul
            if t != max_steps - 1:
                next_obs = obs_
            else:
                next_obs = None
            # sum des rexards pour tous les agents
            total_reward += reward.sum().cpu().item()

            # stockage de la transition
            maddpg.memory.store(obs.data, action, next_obs, reward)
            # on reprend l'ancienne valeur
            obs = next_obs

            maddpg.train()
        maddpg.episode_count += 1
        logger.direct_write("reward", total_reward, i_episode)
        print(str(i_episode) + " rsum=" + str(total_reward) )

        mean += total_reward


    env.close()
