import numpy as np
import collections
import torch
import random

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

class Buffer():
    """
    version du buffer qui renvoie un tensor pas une liste
    """
    def __init__(self):
        self.buffer = collections.deque(maxlen=80000)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done, info = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if (done and not info.get('TimeLimit.truncated', False) ) else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)
