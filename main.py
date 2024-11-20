import gymnasium as gym
import torch
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from utils import plot_metrics, save_model, load_model , epsilon_decay_schedule # Optional utility functions


def train(agent, env, replay_buffer, num_episodes):
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    target_update_freq = 10
    batch_size = 64

    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)  # Ensure state is a tensor
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)  # Ensure next_state is a tensor
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size * 50:
                agent.optimize_model(batch_size, gamma=0.99)

        epsilon = epsilon_decay_schedule(epsilon, epsilon_min, epsilon_decay)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# def evaluate():


# def parse_arguments():


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    replay_buffer = ReplayBuffer(capacity=10000)
    agent = DQNAgent(state_dim=8, action_dim=4, replay_buffer=replay_buffer)
    # Run the training
    train(agent, env, replay_buffer, num_episodes=773)
