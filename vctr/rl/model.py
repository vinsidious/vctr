import torch
from torch.functional import F
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

        self.value_head = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.to('mps')

    def forward(self, x):
        x.to('mps')

        x_policy = self.fc1(x)
        x_policy = F.relu(x_policy)
        x_policy = self.fc2(x_policy)
        x_policy = F.relu(x_policy)
        action_probs = self.fc3(x_policy)
        action_probs = F.softmax(action_probs, dim=-1)

        state_values = self.value_head(x)

        return action_probs, state_values
        return x
