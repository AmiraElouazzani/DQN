import matplotlib.pyplot as plt  # For plotting training metrics
import numpy as np
import cv2  # OpenCV library for image processing
import torch

def plot_metrics(episode_rewards, epsilon_values):
    plt.figure(figsize=(12, 5))

    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    # Plot epsilon values
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values, label="Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.show()



def process_frame(frame):
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Resize to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    # Normalize pixel values
    frame = frame / 255.0

    return frame

def epsilon_decay_schedule(epsilon, epsilon_min, epsilon_decay):
    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    return epsilon



def save_model(agent, filename):
    torch.save(agent.q_network.state_dict(), filename)

def load_model(agent, filename):
    agent.q_network.load_state_dict(torch.load(filename))
    agent.q_network.eval()  # Set to evaluation mode
