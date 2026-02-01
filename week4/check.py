import gymnasium as gym
import torch
import torch.nn as nn

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


# environment
env = gym.make("Humanoid-v4", render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
policy.load_state_dict(torch.load("model", map_location=DEVICE))
policy.eval()


state, _ = env.reset()

for _ in range(1000):
    state_t = torch.FloatTensor(state).to(DEVICE)

    with torch.no_grad():
        mean, _ = policy(state_t)
        action = mean.cpu().numpy()

    state, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        break

env.close()
