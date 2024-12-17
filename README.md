# **Deep Q-Learning for Lunar Lander - Game Theory Approach**

This project implements a **Deep Q-Learning (DQN)** agent to play the **Lunar Lander** game using **reinforcement learning**. The implementation is based on the groundbreaking paper **"Playing Atari with Deep Reinforcement Learning"**, which introduced the DQN algorithm.

The environment is provided by **OpenAI Gymnasium** and simulates a spacecraft that must land safely on a platform. The agent must optimize its actions—thrusting, moving left/right, or doing nothing—by balancing **exploration** (trying new strategies) and **exploitation** (using learned strategies to maximize rewards).

---

## **Game Theory in Action**

This project connects reinforcement learning to **game theory concepts**:
- **Agent**: The AI (decision-maker) attempts to land the spacecraft safely.
- **Environment**: Physical forces (gravity, inertia) act as the agent’s "opponent."
- **Rewards/Payoffs**: Feedback to the agent:
   - Positive rewards for safe landing.
   - Negative rewards for crashing or inefficient fuel usage.

The agent’s **decision-making process** involves:
1. Continually exploring new actions.
2. Refining its strategy to maximize long-term rewards, similar to players solving for optimal strategies in **game theory**.

---

## **How It Works**

### 1. **Deep Q-Learning (DQN)**
- DQN combines **Q-Learning** (a traditional reinforcement learning algorithm) with **deep neural networks** to approximate Q-values.
- **Q-values** represent the agent’s expected future rewards for each action in a given state.
- For this project, we use:
   - **3 hidden layers** in the neural network.
   - **ReLU** activation function for non-linear learning.

### 2. **Exploration vs. Exploitation**
- The agent uses an **epsilon-greedy policy**:
   - **Exploration**: Takes random actions with probability \( \epsilon \).
   - **Exploitation**: Chooses the best-known action based on the Q-network predictions.
- **Epsilon Decay**: The epsilon value decays gradually to encourage more exploitation as training progresses, with a lower bound of **0.05** for continuous exploration.

### 3. **Replay Buffer**
- The **Replay Buffer** stores the agent's past experiences \((state, action, reward, next state)\).
- Key benefits:
   - **Breaking Correlations**: Randomly sampling experiences reduces the bias caused by consecutive, highly correlated samples.
   - **Efficient Learning**: Each experience can be used multiple times to improve training stability.

### 4. **Reward System**
The reward system is designed to guide the agent:
- **+100** for successfully landing.
- **-100 or worse** for crashing.
- Smaller penalties for excessive fuel consumption or drifting off-course.
- To further encourage successful landings, an **additional reward of +200** is given upon winning.

---

## **Project Structure**

- **`main.py`**: Main script for training the agent.
- **`play.py`**: Script for evaluating a trained agent over multiple games (e.g., 5 games).
- **`dqn_agent.py`**: Defines the `DQNAgent` class for decision-making, learning, and Q-value optimization.
- **`replay_buffer.py`**: Implements the `ReplayBuffer` for experience storage and sampling.
- **`network.py`**: Neural network architecture for predicting Q-values.
- **`utils.py`**: Helper functions for saving/loading models, epsilon decay, and other utilities.

---

## **Key Features of the Agent**

1. **Strategic Decision-Making**
   - The agent predicts the **Q-values** for all possible actions using its neural network and selects the best one based on its policy.

2. **Reward System**
   - Reward structure:
     - **+100** for landing.
     - **-100 or worse** for crashes.
     - **+200** additional reward for favoring successful landings.
   - The reward system strongly biases the agent toward **safe and efficient landings**.

3. **Balancing Exploration and Exploitation**
   - The epsilon-greedy policy starts with high exploration (\( \epsilon = 1.0 \)) and decays over time.
   - The decay rate ensures a smooth transition from exploration to exploitation, with a minimum epsilon value of \( 0.05 \).

---

## **Results**

After training for **2,000 episodes**, the agent typically converges to stable strategies:

- For **epsilon = 0.05**:
   - The agent develops clear strategies, such as descending while drifting slightly to one side, then correcting abruptly just before landing.
   - This behavior has a **high success rate**.

- At earlier stages (or higher epsilon):
   - The agent often exhibits exploratory behaviors, such as hovering, wasting fuel, or attempting risky maneuvers.
   - Occasionally, the agent lands next to the platform and performs small thrusts to nudge itself onto the target—illustrating a creative, albeit inefficient, strategy.

---

## **Running the Project**

### **1. Prerequisites**
- Python 3.10+
- Install the required libraries:
   ```bash
   pip install torch gymnasium matplotlib

### **2. Training**
   python3 train.py

### **3. Playing**
   python3 play.py

