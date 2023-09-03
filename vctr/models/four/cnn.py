import torch
import torch.nn as nn
import torch.nn.functional as F
from vctr.models.four.thresh import ThresholdManager


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        return torch.matmul(attention_weights, v)


class CNNModel(nn.Module):
    def __init__(self, num_features, num_classes, conv_channels, dropout_rate, kernel_size, attention=True):
        super(CNNModel, self).__init__()
        self.thresh = ThresholdManager(num_classes)
        self.thresholds = [0, 0, 0]
        
        self.num_features = num_features
        self.attention = attention

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.conv1 = nn.Conv1d(num_features, conv_channels, kernel_size, padding=kernel_size // 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.batch_norm2 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.batch_norm3 = nn.BatchNorm1d(conv_channels)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.maxpool = nn.MaxPool1d(kernel_size, stride=kernel_size // 2, padding=kernel_size // 2)

        if self.attention:
            self.attention_layer = SelfAttention(conv_channels)

        self.fc = nn.Linear(conv_channels, num_classes)

        self.to('mps')

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.transpose(1, 2)  # shape: (batch_size, num_features, sequence_length)

        x = self.batch_norm1(x)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = self.batch_norm2(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = self.batch_norm3(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)

        x = self.maxpool(x)

        if self.attention:
            x = x.transpose(1, 2)  # shape: (batch_size, sequence_length, conv_channels)
            x = self.attention_layer(x)
            x = x.transpose(1, 2)  # shape: (batch_size, conv_channels, sequence_length)

        x = x.mean(dim=-1)  # Global Average Pooling
        x = self.fc(x)
        return x
