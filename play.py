import gymnasium as gym
import torch
from dqn_agent import DQNAgent
from utils import load_model

def play(model_path, num_episodes=5):
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = DQNAgent(state_dim=8, action_dim=4, replay_buffer=None)
    load_model(agent, model_path)
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # the agent will always choose the best-known action based on the Q-values learned during training
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    model_path = "trained_model.pth"  # Path to your trained model
    play(model_path, num_episodes=5)
