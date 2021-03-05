from Actor_critic import Q, Mu
import torch
from copy import deepcopy
from memory import Buffer
from torch.optim import Adam
import torch.nn as nn
import numpy as np


class MADDPG:
    def __init__(self, nbAgents, input_dim, dim_act, batch_size,
                 capacity, eploration_phase):
        # compteurs
        self.steps_done = 0
        self.episode_count = 0
        # ============= parameters =======================

        self.nbAgents = nbAgents
        self.NbEtats = input_dim
        self.nbActions = dim_act
        #buffer
        self.memory = Buffer(capacity)
        # batch_size
        self.batch_size = batch_size
        self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # number of episodess before traininng
        self.eploration_phase = eploration_phase
        # hard coded
        self.gamma = 0.95
        self.ro = 0.01
        # liste des Mus et Qs
        self.Mus = [Mu(input_dim, dim_act) for i in range(nbAgents)]
        self.Qs = [Q(nbAgents, input_dim,
                               dim_act) for i in range(nbAgents)]
        # les cibles
        self.Mus_target = deepcopy(self.Mus)
        self.Qs_target = deepcopy(self.Qs)
        # poids du bruit sur les mu lors du choix de l'action
        self.random_weight = [1.0 for i in range(nbAgents)]
        # liste d'optimizeurs
        self.Qs_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.Qs]
        self.Mus_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.Mus]

        #transferrer sur GPU
        self.transfer_to_device()


    def transfer_to_device(self):
        """
        fonction pour transferrer sur GPU
        """
        # les Mus
        for x in self.Mus:
            x.to(self.device)
        # les Qs
        for x in self.Qs:
            x.to(self.device)
        # les Mus cible
        for x in self.Mus_target:
            x.to(self.device)
        # les Q cibles
        for x in self.Qs_target:
            x.to(self.device)

    def update(self, target, source, ro):
        """
        mise a jour des reseaux selon un poids ro
        """
        for target_param, source_param in zip(target.parameters(),
                                              source.parameters()):
            target_param.data.copy_(
                (1 - ro) * target_param.data + ro * source_param.data)

    def choose(self, state_batch):
        """
        fonction pour choisir les actions
        """
        # state_batch: nbAgents x state_dim
        actions = torch.zeros(self.nbAgents,self.nbActions)
        # pour chaque agent
        # on tire selon le Mu
        for i in range(self.nbAgents):
            sb = state_batch[i, :].detach()
            # transormer en batch de taille 1 avant de recompresser
            act = self.Mus[i](sb.unsqueeze(0)).squeeze()
            # rajouter un bruit gaussien avec un poids
            act += torch.from_numpy(
                np.random.randn(2) * self.random_weight[i]).float().to(self.device)
            # mettre a jour le poids
            # si dans la phase d'entrainement er on a atteint le minimum
            if self.episode_count > self.eploration_phase and self.random_weight[i] > 0.05:
                self.random_weight[i] *= 0.999998
            # max des valeurs
            act = torch.clamp(act, -1.0, 1.0)
            # les actions
            actions[i, :] = act
        self.steps_done += 1

        return actions

    def train(self):
        """
        training function
        """
        # leave a few steps tp explore
        if self.episode_count <= self.eploration_phase:
            return

        # pour chaque agent on sample
        for agent in range(self.nbAgents):
            # batch par agent
            state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = self.memory.sample(self.batch_size)

            # for current agent
            tout_etats = state_batch.view(self.batch_size, -1)
            tout_actions = action_batch.view(self.batch_size, -1)
            # mettre le grad de cet a gent a 0
            self.Qs_optimizer[agent].zero_grad()
            # valeur de Q(etats, actions)
            actual_value_Q = self.Qs[agent](tout_etats, tout_actions)
            # valeur de Mu pour les etats dont l'action suivante n'est pas finale
            non_final_next_actions = torch.stack([self.Mus_target[i](non_final_next_states[:,i,:]) for i in range(self.nbAgents)])
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())
            # target initialisation
            TD_0 = torch.zeros(self.batch_size).float().to(self.device)
            # mettre a zeros les etats finaux
            # et la valeur de Q target pour les autres
            TD_0[non_final_mask] = self.Qs_target[agent](non_final_next_states.view(-1, self.nbAgents * self.NbEtats),non_final_next_actions.view(-1,self.nbAgents * self.nbActions)).squeeze()
            # TD_0
            TD_0 = (TD_0.unsqueeze(1) * self.gamma) + (reward_batch[:, agent].unsqueeze(1) * 0.01)
            # MSE loss comme dans l'algo (d'habitude on utilise huber loss)
            loss_Q = nn.MSELoss()(actual_value_Q, TD_0.detach())
            #======================
            # pas de gradient
            #======================
            loss_Q.backward()
            self.Qs_optimizer[agent].step()
            #======================
            # mise a jour du Mu
            #======================
            self.Mus_optimizer[agent].zero_grad()
            # seulement les mus pour cet agent
            state_i = state_batch[:, agent, :]
            action_i = self.Mus[agent](state_i)
            # remplacement par les valeurs de mu target pour cet agent et les memes actions pour les autres
            ac = action_batch.clone()
            # on remplace juste pour cet agent
            ac[:, agent, :] = action_i
            # mise en forme pour la loss
            tout_actions = ac.view(self.batch_size, -1)
            # comme defini
            mu_loss = -self.Qs[agent](tout_etats, tout_actions)
            # backward
            mu_loss = mu_loss.mean()
            mu_loss.backward()
            self.Mus_optimizer[agent].step()


        # mettre a jour toutes les 100 episodes
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            # pour chaque agent : mise a jour des q et mu
            for i in range(self.nbAgents):
                self.update(self.Qs_target[i], self.Qs[i], self.ro)
                self.update(self.Mus_target[i], self.Mus[i], self.ro)

        return
