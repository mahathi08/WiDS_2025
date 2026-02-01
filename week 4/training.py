import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# =====================
# Settings
# =====================
ENV_NAME = "Humanoid-v4"
EPISODES = 500
MAX_STEPS = 1000
GAMMA = 0.99
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class REINFORCE:
    def __init__(self, obs_dim, act_dim):
        self.policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(DEVICE)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.cpu().numpy(), log_prob

    def update(self, rewards, log_probs):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = REINFORCE(obs_dim, act_dim)
    reward_history = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        rewards = []
        log_probs = []
        total_reward = 0

        for _ in range(MAX_STEPS):
            action, log_prob = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            total_reward += reward

            if terminated or truncated:
                break
        agent.update(rewards, log_probs)
        reward_history.append(total_reward)

        if ep % 10 == 0:
            print(f"Episode {ep} | Reward: {total_reward:.2f}")
    env.close()

    # Save model
    torch.save(agent.policy.state_dict(), "model")
    print("Model saved as model")

    # Plot
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE on Humanoid")
    plt.show()


train()
