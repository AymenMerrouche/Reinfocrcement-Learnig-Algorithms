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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from explorer import *
from memory import *
from utils import *
import pdb





class PPO_clipped(nn.Module):
    def __init__(self, env, opt, alpha, gamma, lmbda, K, epsilon ):
        super(PPO_clipped, self).__init__()
        # buffer customisé pour clipped car on rajoute les proba de l'action
        self.buffer = Buffer_clipped() # rajouté dans memory
        # params
        self.alpha, self.gamma, self.lmbda, self.K, self.epsilon = alpha, gamma, lmbda, K, epsilon
        # action space et featureExtractor
        self.opt, self.env = opt, env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        #==================================================================================
        #                            parametres
        #==================================================================================
        # Comme dans Actor critic : utilisation d'un linear patagé entre pi et v (ca marche mieux de cette maniere)

        self.linear1   = nn.Linear(self.featureExtractor.outSize,256)
        # deuxieme couche pour pi
        self.linear_policy = nn.Linear(256,self.action_space.n)
        # deuxieme couche pour v
        self.linear_value  = nn.Linear(256,1)
        # un seul optimizeur pour les deux ( de ce que je vois, ca se fait couremement)
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

    def advantage(self, difference):
        """ fonction qui calcul les avantages a partir du tensor difference :  td(0) - value"""
        avantages, adv = [], 0.0
         # on inverse les differences car comme ca on calcul dans l'autre sens
        for difference_t in difference[::-1]:
            #le [0] pcq batch
            adv = difference_t[0] +  self.gamma * self.lmbda * adv
            avantages.append([adv])
        # inverser les avantages car calulés a l'envers
        avantages.reverse()
        # tensorisation
        avantages = torch.tensor(avantages, dtype=torch.float)
        return avantages


    def train(self):
        # sampling des trajectoires collectés du buffer
        #pdb.set_trace()
        s, a, r, s_prime, dones, prob_a = self.buffer.sample()
        # pour chaque K
        for i in range(self.K):

            # td(0) pour v
            td_0 = r + self.gamma * self.V_phi(s_prime) * dones

            difference = td_0 - self.V_phi(s)
            difference = difference.detach().numpy()
            # calcul des avantages
            avantages = self.advantage(difference)
            # calcul des probas
            pi = self.PI_theta(s, batch = True)
            # selectionne ceux de l'action a (tensor)
            pi_a = pi.gather(1,a)
            # exp log (a -b) <=> a /b on fait ca pour eviter les 0
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            # loss combinés des deux v_phi et pi_theta
            # float sur td_0 a cause de lunar bizarement il renvoit float64
            loss = -torch.min(ratio * avantages, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * avantages) + F.smooth_l1_loss(self.V_phi(s) , td_0.detach().float())
            #optimization
            self.optimizer.zero_grad()
            # mean :
            loss.mean().backward()
            self.optimizer.step()


    def PI_theta(self, x, batch = False):
        """
        fonction qui calclul la proba de chaque action (comme dans le actor critic)
        """
        x =  self.featureExtractor.getFeatures(x).float()
        x = F.relu(self.linear1(x))
        x = self.linear_policy(x)
        if batch : # si c'est un batch d'exemple ou pas
            prob = F.softmax(x, dim=1)
        else :
            prob = F.softmax(x, dim=0)
        return prob

    def V_phi(self, x):
        """
        fonction qui calclul la valeur d'un etat
        """
        x =  self.featureExtractor.getFeatures(x).float()
        x = F.relu(self.linear1(x))
        v = self.linear_value(x)
        return v


if __name__ == '__main__':

    #config = load_yaml('./configs/config_random_gridworld.yaml')
    #config = load_yaml('./configs/config_random_cartpole.yaml')
    config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    t_step = config["t_step"]

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + f"/PPO_clipped_mean/"+ tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    trains = []
    tests = []

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    Test = False
    # test sur plusieurs run et puis moyennage
    for essai in range(2):

        print("===========================================================")
        print("                      test" + str(essai) )
        print("===========================================================")
        tests.append([])
        trains.append([])

        agent = PPO_clipped(env, config, config["alpha"], config["gamma"], config["lmbda"], config["K"], config["epsilon"])

        # just to check si c'est bon
        pdb.set_trace()

        for i in range(episode_count):
             # on log a la fin quand on moyenne
             # ici on stock dans les list trains et tests
            if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
                print("Test time! ")
                mean = 0
                Test = True

            if i % freqTest == nbTest and i > freqTest:
                print("End of test, mean reward=", mean / nbTest)
                itest += 1
                tests[-1].append(mean / nbTest)
                #logger.direct_write("rewardTest", mean / nbTest, itest)
                Test = False

            done = False
            s = env.reset()

            while not done:
                # on fait t_step pas puis on optimize
                for t in range(t_step):
                    # on tire de la politique l'action
                    prob = agent.PI_theta(torch.from_numpy(s).float())

                    m = Categorical(prob)
                    # on tire selon la distribution
                    a = m.sample().item()

                    s_prime, r, done, info = env.step(a)

                    # on stocke la transition
                    agent.buffer.store((s, a, r/100.0, s_prime, prob[a].item(), done, info))
                    # nouvel etat devient ancien etat
                    s = s_prime
                    rsum += r
                    # si on s'arrete en plein t_step
                    if done:
                        # on arrete la boucle for t_step et on log dans les tests
                        print(str(i) + " rsum=" + str(rsum) )
                        trains[-1].append(rsum)
                        #logger.direct_write("reward", rsum, i)
                        agent.nbEvents = 0
                        agent.episode = i
                        mean += rsum
                        rsum = 0
                        break

                # tester si ca aide de plafonner et arreter l'entrainement
                # si on atteint 500 pour cart
                agent.train()
    env.close()
    pdb.set_trace()
    trains = np.array(trains)
    tests = np.array(tests)
    pdb.set_trace()
    trains_means = trains.mean(axis=0)
    test_means = tests.mean(axis=0)
    for i in range( len(trains_means) ):
        logger.direct_write("reward", trains_means[i], i)
    for i in range( len(test_means) ):
        logger.direct_write("rewardTest", test_means[i], i)
