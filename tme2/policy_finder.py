import matplotlib
import pdb;
#matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns

def valueIteration(stateDic, mdp, epsilon = 0.01, gamma= 0.95):
    V = {state: np.random.rand() if state in mdp.keys() else 0 for state in stateDic.keys()}
    NotOptimal = True
    niter = 0
    while(NotOptimal):
        niter+=1
        v = V.copy()
        for s in mdp.keys():
            V[s] = max([sum([proba * (reward + gamma * v[futureState]) for proba, futureState, reward, done in mdp[s][a]]) for a in mdp[s]])
        if np.linalg.norm(np.array(list(v.values()))-np.array(list(V.values()))) < epsilon:
            NotOptimal = False
    pi = {}
    for state in mdp.keys():
        pi[state] = np.argmax( [ sum([proba * (reward + gamma * V[futureState]) for proba, futureState, reward, done in mdp[state][a]]) for a in mdp[state]])
    return pi, V, niter

def policyIteration(statedic, mdp, epsilon= 0.01, gamma = 0.95):
    # initialise
    PI ={ state : np.random.randint(0,4) for state in mdp.keys()}
    optimal = False
    niter= 0
    while(not optimal):
        V = {state: np.random.rand() if state in mdp.keys() else 0  for state in statedic.keys()}
        converge = False
        while(not converge):
            niter +=1
            v = V.copy()
            for state in mdp:
                V[state] = sum( [ proba * (reward + gamma * v[futureState]) for proba, futureState, reward, done in mdp[state][PI[state]] ] )
            if (np.linalg.norm( np.array(list(v.values())) -np.array(list(V.values())) ) < epsilon):
                converge = True
        pi = PI.copy()
        for state in mdp:
            PI[state] = np.argmax( [ sum([proba * (reward + gamma * V[futureState]) for proba, futureState, reward, done in mdp[state][a]]) for a in mdp[state]])
        if pi == PI:
            optimal = True
    return PI, V, niter



def test():
    rewards = {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1}
    policyIt, valueIt = [],[]
    policyNIt, valueNIt = [],[]
    print("Value iteration")
    for i in [0,1,2,3,4,5,6,7,8,10]:
        print("plan ", i)
        env = gym.make("gridworld-v0")
        env.setPlan("gridworldPlans/plan" +str(i)+  ".txt", rewards)

        env.seed(0)  # Initialise le seed du pseudo-random
        statedic, mdp = env.getMDP()  # recupere le mdp : statedic
        a = time.time()
        pi, V, niter = valueIteration(statedic, mdp , epsilon= 0.01)
        valueIt.append(time.time() - a)
        valueNIt.append(niter)
    print("policyIt")
    for i in [0,1,2,3,4,5,6,8,10]:
        print("plan ", i)
        env = gym.make("gridworld-v0")
        env.setPlan("gridworldPlans/plan" +str(i)+  ".txt", rewards)

        env.seed(0)  # Initialise le seed du pseudo-random
        statedic, mdp = env.getMDP()  # recupere le mdp : statedic
        a = time.time()
        pi, V , niter= policyIteration(statedic, mdp , epsilon= 0.01, gamma = 0.95)
        policyIt.append(time.time() - a)
        policyNIt.append(niter)
        print("done")
        env.close()
    return policyIt, valueIt ,policyNIt, valueNIt

"""sns.set()
plt.bar([0,1,2,3,4,5,6,8,10], policyIt)
plt.bar([0,1,2,3,4,5,6,7,8,10], valueIt )
plt.legend(["Policy Iteration", "Value iteration"])
plt.xlabel("plan")
plt.ylabel("time")
#plt.title("Comparaison du temps de convergence des deux algorithmes")
plt.savefig("./graphs/time_convergence.png")
plt.clf()
plt.bar([0,1,2,3,4,5,6,8,10], policyNIt)
plt.bar([0,1,2,3,4,5,6,7,8,10], valueNIt )
plt.legend(["Policy Iteration", "Value iteration"])
plt.xlabel("plan")
plt.ylabel("time")
plt.savefig("./graphs/iteration_convergence.png")
"""
