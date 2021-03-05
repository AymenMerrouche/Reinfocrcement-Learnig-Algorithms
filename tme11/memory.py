from collections import namedtuple
import random
import torch
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))


class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device  = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def store(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        batch = Experience(*zip(*transitions))
        non_final_mask = torch.tensor(list(map(lambda s: s is not None,
                                             batch.next_states))).bool().to(self.device)
        # state_batch: batch_size x nbAgents x input_dim
        state_batch = torch.stack(batch.states).float().to(self.device)
        action_batch = torch.stack(batch.actions).float().to(self.device)
        reward_batch = torch.stack(batch.rewards).float().to(self.device)
        # : (batch_size_non_final) x nbAgents x input_dim
        non_final_next_states = torch.stack(
            [s for s in batch.next_states
             if s is not None]).float().to(self.device)
        return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states

    def __len__(self):
        return len(self.memory)
