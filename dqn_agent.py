import torch
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F  
import numpy as np  
import random
from network import QNetwork 

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_dim))
        else:
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def optimize_model(self, batch_size, gamma=0.99):
        #Get the batch from the replay buffer instance
        batch = self.replay_buffer.sample(batch_size)

        QValues = self.q_network(batch[0])  # Predicted Q-values for all actions
        Actions = torch.reshape(batch[1], [batch_size, 1])
        PredictedQValues = QValues.gather(1, Actions).squeeze(1)  # Shape [batch_size]

        #max Q-value for the next states using the target network
        NextQValues = self.target_network(batch[3])
        MaxNextQValues = torch.max(NextQValues, dim=1)[0]

        rewards = batch[2]
        dones = batch[4].float()

        TargetQValues = rewards + gamma * MaxNextQValues * (1 - dones)
        TargetQValues = TargetQValues.detach()

        #loss between predicted Q-values and target Q-values
        loss = torch.nn.functional.smooth_l1_loss(PredictedQValues, TargetQValues)

        # Backpropagation to update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state, epsilon):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if torch.rand(1).item() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                return self.q_network(state).argmax().item()





    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
