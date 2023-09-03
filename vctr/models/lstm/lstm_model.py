import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


import torch
import torch.nn as nn
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = self.fc(x).squeeze(2)
        weights = F.softmax(weights, dim=1)
        weights = weights.unsqueeze(2)
        weighted_sequence = x * weights
        out = weighted_sequence.sum(dim=1)
        return out


class HybridModel(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size,
        num_layers,
        num_classes,
        bidirectional,
        kernel_size,
        out_channels,
        dropout=0.2,
        use_conv=True,  # Add use_conv parameter
    ):
        super(HybridModel, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_conv = use_conv  # Store the use_conv parameter

        # Convolutional Layer
        self.conv1 = nn.Conv1d(num_features, out_channels, kernel_size)
        self.dropout1 = nn.Dropout(dropout)

        # LSTM Layer
        self.lstm = nn.LSTM(
            num_features if not use_conv else out_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        # Attention Layer
        self.attention = Attention(hidden_size * self.num_directions)

        # Fully connected Layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        self.to('mps')

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # x shape: (batch_size, num_features, sequence_length)

        # Apply Conv1D layer if use_conv is True
        if self.use_conv:
            x = self.conv1(x)  # x shape: (batch_size, out_channels, new_sequence_length)
            x = self.dropout1(x)
            x = x.permute(0, 2, 1)  # x shape: (batch_size, new_sequence_length, out_channels)
        else:
            x = x.permute(0, 2, 1)  # x shape: (batch_size, sequence_length, num_features)

        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(
            x, (h0, c0)
        )  # out shape: (batch_size, sequence_length or new_sequence_length, hidden_size * num_directions)

        # Apply Attention layer
        out = self.attention(out)

        # Apply Fully Connected layer
        out = self.fc(out)  # out shape: (batch_size, num_classes)

        return out


class GRUConvNetModel(nn.Module):
    def __init__(
        self, num_features, hidden_size, num_layers, num_classes, bidirectional, kernel_size, out_channels, dropout
    ):
        super(GRUConvNetModel, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(num_features, out_channels, kernel_size)
        self.dropout1 = nn.Dropout(dropout)

        # GRU Layer
        self.gru = nn.GRU(
            out_channels, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )

        # Fully connected Layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        self.to('mps')

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # x shape: (batch_size, num_features, sequence_length)

        # Apply Conv1D layer
        x = self.conv1(x)  # x shape: (batch_size, out_channels, new_sequence_length)
        x = self.dropout1(x)

        # Apply GRU layer
        x = x.permute(0, 2, 1)  # x shape: (batch_size, new_sequence_length, out_channels)
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)  # out shape: (batch_size, new_sequence_length, hidden_size * num_directions)

        # Apply Fully Connected layer
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, num_classes)

        return out


class CNN_GRU(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_classes, kernel_size, out_channels, dropout):
        super(CNN_GRU, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Convolutional Layer
        self.conv1 = nn.Conv1d(num_features, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)  # Batch Normalization
        self.relu = nn.ReLU()

        # GRU Layer
        self.gru = nn.GRU(
            out_channels, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected Layer
        self.fc = nn.Linear(hidden_size, num_classes)

        self.to('mps')

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # x shape: (batch_size, num_features, sequence_length)

        # Apply Conv1D layer
        x = self.conv1(x)  # x shape: (batch_size, out_channels, new_sequence_length)
        x = self.bn(x)
        x = self.relu(x)

        # Apply GRU layer
        x = x.permute(0, 2, 1)  # x shape: (batch_size, new_sequence_length, out_channels)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)  # out shape: (batch_size, new_sequence_length, hidden_size)

        # Apply Dropout
        out = self.dropout(out)

        # Apply Fully Connected layer
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, num_classes)

        return out


class CNNGRU(nn.Module):
    def __init__(
        self, num_features, hidden_size, num_layers, num_classes, bidirectional, kernel_size, out_channels, dropout
    ):
        super(CNNGRU, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(num_features, out_channels, kernel_size)
        self.conv1_bn = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        # GRU Layer
        self.gru = nn.GRU(
            out_channels, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )

        # Fully connected Layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        self.to('mps')

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # x shape: (batch_size, num_features, sequence_length)

        # Apply Conv1D layer
        x = self.conv1(x)  # x shape: (batch_size, out_channels, new_sequence_length)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Apply GRU layer
        x = x.permute(0, 2, 1)  # x shape: (batch_size, new_sequence_length, out_channels)
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)  # out shape: (batch_size, new_sequence_length, hidden_size * num_directions)
        out = out[:, -1, :]  # out shape: (batch_size, hidden_size * num_directions)

        # Apply Fully Connected layer
        out = self.fc(out)  # out shape: (batch_size, num_classes)

        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_classes, num_layers=3, num_heads=4, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(num_features, dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_features, nhead=num_heads, dropout=dropout),
            num_layers=num_layers,
        )

        self.decoder = nn.Linear(num_features, num_classes)

        self.init_weights()

        self.to('mps')

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (batch_size, sequence_length, num_features)
        src = src.permute(1, 0, 2)  # permuted src shape: (sequence_length, batch_size, num_features)

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)

        # Only take the class probabilities of the last time step
        output = output[-1, :, :]

        return output  # output shape: (batch_size, num_classes)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.to('mps')

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, 'Embedding size needs to be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.to('mps')

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Calculate the dot product between the queries and keys to get the similarity matrix
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

        self.to('mps')

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_layers,
        embed_size,
        num_heads,
        dropout,
        forward_expansion,
    ):
        super(Transformer, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.embed_size = embed_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(num_features, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, num_classes)

        self.to('mps')

    def forward(self, x):
        N, T, E = x.size()
        out = self.embedding(x)

        for layer in self.layers:
            out = layer(out, out, out)

        out = self.fc_out(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, num_heads, num_classes, dropout):
        super(TransformerModel, self).__init__()

        self.thresholds = [0, 0.85, 0.85]

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding2(num_features, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_features, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(num_features, num_classes)

        self.init_weights()

        self.to('mps')

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)

        # Apply softmax and get the most likely class
        output = torch.nn.functional.softmax(output, dim=-1)
        output = torch.argmax(output, dim=-1)

        return output


class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.to('mps')

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
