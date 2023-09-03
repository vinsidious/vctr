import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vctr.rl.model import PolicyNetwork
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.clip_epsilon = 0.2

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device='mps')
        action_probs, _ = self.policy(state_tensor)  # Unpack the tuple returned by the policy network
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def train(self, states, actions, log_probs_old, returns, advantages, update_epochs=10):
        states_tensor = torch.tensor(states, dtype=torch.float32, device='mps')
        actions_tensor = torch.tensor(actions, dtype=torch.long, device='mps')
        log_probs_old_tensor = torch.tensor(log_probs_old, dtype=torch.float32, device='mps')
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device='mps')
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device='mps')

        for _ in range(update_epochs):  # Number of epochs for PPO updates
            action_probs, state_values = self.policy(states_tensor)
            distribution = Categorical(action_probs)
            log_probs_new = distribution.log_prob(actions_tensor)
            entropy = distribution.entropy().mean()

            ratio = torch.exp(log_probs_new - log_probs_old_tensor)
            surrogate1 = ratio * advantages_tensor
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor

            # Add the value function loss (mse_loss) to the total loss
            loss = (
                -torch.min(surrogate1, surrogate2)
                + 0.5 * nn.functional.mse_loss(returns_tensor, state_values.squeeze())
                - 0.01 * entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
