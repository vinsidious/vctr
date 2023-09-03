import torch
import torch.nn as nn
import torch.optim as optim


class HybridRNNLSTM(nn.Module):
    def __init__(
        self, input_size, rnn_hidden_size, lstm_hidden_size, num_classes, rnn_layers=1, lstm_layers=1, dropout=0.5
    ):
        super(HybridRNNLSTM, self).__init__()
        self.thresholds = [0, 0, 0]

        self.rnn = nn.RNN(input_size, rnn_hidden_size, num_layers=rnn_layers, batch_first=True)
        self.lstm = nn.LSTM(rnn_hidden_size, lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

        self.to('mps')

    def forward(self, x):
        # Pass input through RNN layer
        out, _ = self.rnn(x)

        # Pass RNN output through LSTM layer
        out, _ = self.lstm(out)

        # Take the output of the last time step
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # Pass output through the fully connected layer for classification
        out = self.fc(out)
        return out
