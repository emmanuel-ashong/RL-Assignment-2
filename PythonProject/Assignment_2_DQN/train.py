import gymnasium as gym
import torch
import torch.optim as optim
import random
import numpy as np
import pandas as pd  # Added to save your data
from models import QNetwork, ReplayBuffer

# --- 1. SETUP ---
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = 2  # CartPole always has 2 actions

# Hyperparameters
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995
TOTAL_STEPS = 1000000  # The 10^6 requirement

# --- 2. INITIALIZE ---
policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# Data Tracking
results = []  # To store [step, reward]
steps_done = 0
epsilon = EPS_START

# --- 3. TRAINING LOOP ---
print("Starting training...")

while steps_done < TOTAL_STEPS:
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Select Action (Epsilon-Greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_t).argmax().item()

        # Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store & Update
        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        steps_done += 1

        # Epsilon Decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Optimization Step
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, n_states, dones = memory.sample(BATCH_SIZE)

            # DQN Loss Logic (Task 1.1)
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                max_next_q = policy_net(n_states).max(1)[0]
                target_q = rewards + (GAMMA * max_next_q * (1 - dones))

            loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sync Target Network
        #if steps_done % TARGET_UPDATE == 0:
            #target_net.load_state_dict(policy_net.state_dict())

    # Save data every episode
    results.append([steps_done, episode_reward])

    # Progress Print
    if len(results) % 20 == 0:
        print(f"Step: {steps_done} | Last Reward: {episode_reward} | Eps: {epsilon:.2f}")

# --- 4. SAVE DATA TO CSV ---
df = pd.DataFrame(results, columns=['step', 'reward'])
df.to_csv('only_er_5.csv', index=False)
print("Training Complete! Data saved to 'only_er_5.csv'")

env.close()