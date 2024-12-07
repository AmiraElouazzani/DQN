import random
import numpy as np
from collections import deque
import torch
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity #max size of stored experience
        self.buffer = deque(maxlen=capacity)

    #adding experience to deque
    def push(self, state, action, reward, next_state, done):
        # Ensure states and next_states are added as tensors
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(next_state, np.ndarray):
            next_state = torch.tensor(next_state, dtype=torch.float32)

        # experience to buffer
        if len(self.buffer) >= self.capacity:
            self.buffer = deque(maxlen=self.capacity)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # Unpack experiences + convert each component to a tensor
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Stack to create a single batch tensor
        states = torch.stack(states)
        next_states = torch.stack(next_states)

        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
