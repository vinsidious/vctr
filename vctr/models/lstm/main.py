import torch.nn as nn
from torch.functional import F


class LSTMConv1d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, padding=1):
        super(LSTMConv1d, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.transpose(1, 2)  # Convert to (batch, hidden_size, seq_len)
        out = self.conv(out)
        out = out.transpose(1, 2)  # Convert back to (batch, seq_len, hidden_size)
        return out


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.ModuleList(
            [LSTMConv1d(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.to('mps')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.fc(x[:, -1, :])
        out = self.softmax(out)
        return out


class VCNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_layers,
    ):
        super(VCNet, self).__init__()
        self.thresholds = [0, 0.85, 0.85]

        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([LSTMConv1d(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, num_classes)

        self.to('mps')

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]

        action_probs = self.fc(x)
        action_probs = F.softmax(action_probs, dim=-1)

        return action_probs
