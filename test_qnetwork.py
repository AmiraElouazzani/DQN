import torch
from network import QNetwork  

#state dimension and action dimension for Lunar Lander
state_dim = 8     
action_dim = 4    

q_network = QNetwork(state_dim, action_dim)

dummy_input = torch.randn(1, state_dim)  #tensor with shape [1, state_dim]

output = q_network(dummy_input)

print("Output shape:", output.shape)  # Expected shape: [1, action_dim]
