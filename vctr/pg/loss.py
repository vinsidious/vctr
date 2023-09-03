import torch
from vctr.pg.model import PolicyNetwork
import torch.optim as optim

# Instantiate the policy network
input_size = 484  # number of features
hidden_size = 128
num_layers = 2
output_size = 3  # number of actions (HOLD, BUY, SELL)
policy = PolicyNetwork(input_size, hidden_size, num_layers, output_size)

# Set the learning rate and create an optimizer
learning_rate = 0.001
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Implement the loss function
def compute_loss(states, actions, rewards_to_go):
    action_probabilities = policy(states)
    action_log_probs = torch.log(action_probabilities)
    selected_action_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    loss = -torch.mean(selected_action_log_probs * rewards_to_go)
    return loss
