import torch


def epsilon_decay_schedule(epsilon, epsilon_min, epsilon_decay):
    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    return epsilon



def save_model(agent, filename):
    torch.save(agent.q_network.state_dict(), filename)

def load_model(agent, filename):
    agent.q_network.load_state_dict(torch.load(filename))
    agent.q_network.eval()  # Set to evaluation mode
