import torch.nn as nn
from torch.functional import F


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PolicyNetwork, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.to('mps')

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
