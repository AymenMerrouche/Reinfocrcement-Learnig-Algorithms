
import argparse
import sys
import matplotlib
from pathlib import Path
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from utils import *
from torch.utils.tensorboard import SummaryWriter
from explorer import *
from memory import *
from utils import *
import pdb

class PPO_kl(nn.Module):
    def __init__(self, env, opt, alpha, gamma, lmbda, K, delta):
        super(PPO_kl, self).__init__()
        # buffer custom car on stock toutes les distributions
        self.buffer = Buffer_kl()
        self.opt, self.env = opt, env
        self.alpha, self.gamma, self.lmbda, self.K, self.delta = alpha, gamma, lmbda, K, delta
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        # on initialise a 1
        self.beta = 1

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

    def compute_KL(self, old_probabilities, probas):
        """
        fonction qui calcul la kl moyenne entre lex vielles probas et la nouvelle
        """
        kl = torch.distributions.kl_divergence(torch.distributions.Categorical(old_probabilities), torch.distributions.Categorical(probas)
        ).mean()
        return kl

    def advantage(self, difference):
        """ fonction qui calcul les avantages à partir du tensor difference :  td(0) - value"""
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

    def PI_theta(self, x, batch = False):
        """
        la policy PI_theta
        """
        x =  self.featureExtractor.getFeatures(x)
        #x = torch.from_numpy(x).float()

        x = F.relu(self.linear1(x))

        x = self.linear_policy(x)
        # si c'est un batch d'exemple ou pas
        if batch :
            prob = F.softmax(x, dim=1)
        else :
            prob = F.softmax(x, dim=0)
        return prob

    def V_phi(self, x):
        """
        la value V_phi
        """
        x =  self.featureExtractor.getFeatures(x).float()

        #x = torch.from_numpy(x).float()

        x = F.relu(self.linear1(x))

        v = self.linear_value(x)

        return v


    def train(self):
        s, a, r, s_prime, done_mask, prob_a, old_probabilities = self.buffer.sample()

        for i in range(self.K):
            # for grid world, list not tensor
            #v_phisprime = torch.cat([ self.V_phi(ex) for ex in s_prime ])
            #td_0 = r + self.gamma * v_phisprime * done_mask
            td_0 = r + self.gamma * self.V_phi(s_prime) * done_mask
            #v_phis = torch.cat([ self.V_phi(ex) for ex in s ])
            #difference = td_0 - v_phis

            difference = td_0 - self.V_phi(s)
            difference = difference.detach().numpy()

            avantages = self.advantage(difference)

            #probas = torch.cat([ self.PI_theta(ex, batch=True) for ex in s ])

            probas = self.PI_theta(s, batch=True)

            probas_a = probas.gather(1,a)

            L_theta = (torch.exp(torch.log(probas_a) - torch.log(prob_a)) * avantages).mean()


            kl = self.compute_KL(old_probabilities,probas )


            # ===================== version grid world (liste de array pas juste un vecteur)
            #loss = -L_theta + self.beta * kl + F.smooth_l1_loss(v_phis , td_0.detach().float())
            # version lunar et cart

            loss = -L_theta + self.beta * kl + F.smooth_l1_loss(self.V_phi(s) , td_0.detach().float())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #new_probabilities = torch.cat([ self.PI_theta(ex, =1) for ex in s ])
        new_probabilities = self.PI_theta(s, batch=1)

        d_kl = torch.distributions.kl_divergence(torch.distributions.Categorical(old_probabilities),torch.distributions.Categorical(new_probabilities)).mean()
        if d_kl >= 1.5 * self.delta:
            self.beta *= 2
        elif d_kl <= self.delta / 1.5:
            self.beta /= 2
        return

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
    outdir = "./XP/" + config["env"] + f"/PPO_KL/"+ tstart


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
    # version a plusieurs essaies
    for essai in range(10):
        print("===========================================================")
        print("                      test" + str(essai) )
        print("===========================================================")
        tests.append([])
        trains.append([])
        agent = PPO_kl(env, config, config["alpha"], config["gamma"], config["lmbda"], config["K"], config["delta"] )

        for i in range(episode_count):
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
                for t in range(t_step):
                    #  on fait t_step pas puis on optimize
                    prob = agent.PI_theta(torch.from_numpy(s).float())
                    # on tire de la politique l'action
                    m = Categorical(prob)
                    # on tire selon la distribution
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a)
                    # on stocke la transition
                    # on divise par 100, je l'ai vu souvent, car marche mieux avec,
                    # peut etre que c'est pour standardiser et ca entraine plus vite
                    agent.buffer.store((s, a, r/100.0, s_prime, prob[a].item(), prob.tolist(), done, info))
                    # si on s'arrete en plein t_step
                    s = s_prime
                    rsum += r
                    if done:
                        # on arrete la boucle for t_step et on log dans les tests
                        print(str(i) + " rsum=" + str(rsum) )
                        trains[-1].append(rsum)
                        #logger.direct_write("reward", rsum, i)
                        agent.nbEvents = 0
                        agent.episode = i
                        mean += rsum
                        score = rsum
                        rsum = 0
                        break
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
