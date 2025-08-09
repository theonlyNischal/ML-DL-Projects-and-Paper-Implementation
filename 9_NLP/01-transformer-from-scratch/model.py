import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    """
    Embedding layer for input tokens.

    Args:
        input_dim (int): Size of the vocabulary (number of unique tokens).
        embedding_dim (int): Size of each embedding vector (features per token).
    """
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super(InputEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        # The embedding weights are initialized randomly and will be learned during training.
        self.embedding = nn.Embedding(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer to add positional information to input embeddings.

    Args:
        embedding_dim (int): Size of each embedding vector.
        max_len (int): Maximum length of the input sequences.
    """
    def __init__(self, embedding_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

        # Create a matrix of shape (max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]