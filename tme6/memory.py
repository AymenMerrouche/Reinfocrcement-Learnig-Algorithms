import numpy as np
import torch

class SumTree:
    def __init__(self, mem_size):
        self.tree = np.zeros(2 * mem_size - 1)
        self.data = np.zeros(mem_size, dtype=object)
        self.size = mem_size
        self.ptr = 0
        self.nentities=0


    def update(self, idx, p):
        tree_idx = idx + self.size - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def store(self, p, data):
        self.data[self.ptr] = data
        self.update(self.ptr, p)

        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
        self.nentities+=1
        if self.nentities > self.size:
            self.nentities = self.size


    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1

        return ptr - (self.size - 1), self.tree[ptr], self.data[ptr - (self.size - 1)]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def max_p(self):
        return np.max(self.tree[-self.size:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.size:])


class Memory:

    def __init__(self, mem_size, prior=True,p_upper=1.,epsilon=.01,alpha=1,beta=1):
        self.p_upper=p_upper
        self.epsilon=epsilon
        self.alpha=alpha
        self.beta=beta
        self.prior = prior
        self.nentities=0
        #self.data_len = 2 * feature_size + 2
        if prior:
            self.tree = SumTree(mem_size)
        else:
            self.mem_size = mem_size
            self.mem = np.zeros(mem_size, dtype=object)
            self.mem_ptr = 0

    def store(self, transition):
        if self.prior:
            p = self.tree.max_p
            if not p:
                p = self.p_upper
            self.tree.store(p, transition)
        else:
            self.mem[self.mem_ptr] = transition
            self.mem_ptr += 1

            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
            self.nentities += 1
            if self.nentities > self.mem_size:
                self.nentities = self.mem_size
                return False
        return True

    def sample(self, n):
        if self.prior:
            min_p = self.tree.min_p
            if min_p==0:
                min_p=self.epsilon**self.alpha
            seg = self.tree.total_p / n
            batch = np.zeros(n, dtype=object)
            w = np.zeros((n, 1), np.float32)
            idx = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + seg
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.sample(v)

                w[i] = (p / min_p) ** (-self.beta)
                a += seg
            return idx, w, batch
        else:
            mask = np.random.choice(range(self.nentities), n)
            return self.mem[mask]

    def update(self, idx, tderr):
        if self.prior:
            tderr += self.epsilon
            tderr = np.minimum(tderr, self.p_upper)
            #print(idx,tderr)
            for i in range(len(idx)):
                self.tree.update(idx[i], tderr[i] ** self.alpha)


class Buffer_clipped:
    """
    buffer customisé pour la version clipped avec un seul pramaetre prob(a) a stocker
    pas de taille max
    et vider a chaque sampling
    il retourne des tensor directement
    """
    def __init__(self):
        self.buffer = []
        pass
    def store(self, transition):
        """
        stockage d'une transition
        """
        self.buffer.append(transition)

    def sample(self):
        """
        assembler toutes les transitions dans des tensors
        """

        liste_etats, liste_actions, liste_rewards, liste_new_etats,prob_a_lst, liste_done = [], [], [], [], [], []

        for (s,a,r,s_prime,prob_a,done, info) in self.buffer:

            liste_etats.append(s)
            liste_actions.append([a])
            liste_rewards.append([r])
            liste_new_etats.append(s_prime)
            prob_a_lst.append([prob_a])
            # on verifie avec le info.get('TimeLimit.truncated', False) si on s'est arreté car on est nul ou car on a bien fait
            done_mask = 0.0 if (done and not info.get('TimeLimit.truncated', False) )  else 1.0
            liste_done.append([done_mask])

        # assembler le tout dans un tensor
        s_tensor, a_tensor, r_tensor, s_prime_tensor, done_tensor, prob_a = torch.tensor(liste_etats, dtype=torch.float), torch.tensor(liste_actions), \
                                                               torch.tensor(liste_rewards, dtype=torch.float), torch.tensor(liste_new_etats, dtype=torch.float), \
                                                               torch.tensor(liste_done, dtype=torch.float), torch.tensor(prob_a_lst)


        # vider le buffer
        self.buffer = []

        return s_tensor, a_tensor, r_tensor, s_prime_tensor, done_tensor, prob_a

class Buffer_kl:
    """
    buffer customisé pour la version KL
    pas de taille max
    et vider a chaque sampling
    il retourne des tensor directement
    on stock les distributions
    """
    def __init__(self):
        self.buffer = []
        pass
    def store(self, transition):
        self.buffer.append(transition)
    def sample(self):
        liste_etats, liste_actions, liste_rewards, liste_new_etats, prob_a_lst,old_probabilities_lst, done_lst = [], [], [], [], [],[], []
        for (s, a, r, s_prime, prob_a,old_p, done, info) in self.buffer:
            liste_etats.append(s)
            liste_actions.append([a])
            liste_rewards.append([r])
            liste_new_etats.append(s_prime)
            prob_a_lst.append([prob_a])
            old_probabilities_lst.append(old_p)
            done_mask = 0 if (done and not info.get('TimeLimit.truncated', False) ) else 1
            done_lst.append([done_mask])

            s,a,r,s_prime,done_mask, prob_a , old_probabilities= torch.tensor(liste_etats, dtype=torch.float), torch.tensor(liste_actions), \
                                         torch.tensor(liste_rewards), torch.tensor(liste_new_etats, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst),  torch.tensor(old_probabilities_lst)
        self.buffer = []
        return s, a, r, s_prime, done_mask, prob_a, old_probabilities
