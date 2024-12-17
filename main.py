import gymnasium as gym
import torch
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from utils import save_model, load_model , epsilon_decay_schedule

def train(agent, env, replay_buffer, num_episodes):
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update_freq = 10
    batch_size = 64

    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32) 
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            if done:
                if reward >= 100:  # Successful landing
                    reward += 200  
            next_state = torch.tensor(next_state, dtype=torch.float32) 
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            

            if len(replay_buffer) > batch_size * 50:
                agent.optimize_model(batch_size, gamma=0.99)

        epsilon = epsilon_decay_schedule(epsilon, epsilon_min, epsilon_decay)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")


def evaluate(agent, env, num_episodes=1, render_delay=0.5):
    total_rewards = []
    epsilon = 0.0  # No exploration during evaluation

    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        print(f"\n--- Starting Evaluation Episode {episode + 1} ---")

        while not (done or truncated):
            step_count += 1

            env.render() 
            # Select the action greedily
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage Reward over {num_episodes} Evaluation Episodes: {avg_reward:.2f}")
    env.close()
    return avg_reward


if __name__ == "__main__":
    env = gym.make("LunarLander-v2",render_mode="rgb_array")  
    replay_buffer = ReplayBuffer(capacity=10000)
    agent = DQNAgent(state_dim=8, action_dim=4, replay_buffer=replay_buffer)
    
    train(agent, env, replay_buffer, num_episodes=5000)

    save_model(agent, "trained_model.pth")

    load_model(agent, "trained_model.pth")
    avg_reward = evaluate(agent, env, num_episodes=1) 
