"""
RNN models: BiLSTM and BiLSTM + Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config


class BiLSTM(nn.Module):
    """Bidirectional LSTM classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int = config.RNN_EMBEDDING_DIM,
                 hidden_dim: int = config.RNN_HIDDEN_DIM,
                 num_layers: int = config.RNN_NUM_LAYERS,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = config.RNN_DROPOUT,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, emb_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: (num_layers*2, batch, hidden_dim)
        # Concatenate last forward and backward hidden states
        hidden_fwd = hidden[-2]  # (batch, hidden_dim)
        hidden_bwd = hidden[-1]  # (batch, hidden_dim)
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, hidden_dim*2)
        out = self.dropout(hidden_cat)
        logits = self.fc(out)  # (batch, num_classes)
        return logits


class Attention(nn.Module):
    """Additive attention mechanism."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch, seq_len, hidden_dim*2)
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim*2)
        return context, weights


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM + Attention classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int = config.RNN_EMBEDDING_DIM,
                 hidden_dim: int = config.RNN_HIDDEN_DIM,
                 num_layers: int = config.RNN_NUM_LAYERS,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = config.RNN_DROPOUT,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.pad_idx = pad_idx

    def forward(self, x, return_attention: bool = False):
        # x: (batch, seq_len)
        mask = (x != self.pad_idx).float()  # (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        context, attn_weights = self.attention(lstm_out, mask)  # (batch, hidden_dim*2)
        out = self.dropout(context)
        logits = self.fc(out)  # (batch, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits
