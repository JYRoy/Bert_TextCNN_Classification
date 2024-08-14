import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BertConfig:
    vocab_size: int = 65001
    train_max_seq_len: int = 1024
    val_max_seq_len: int = 1024
    max_seq_len: int = max(train_max_seq_len, val_max_seq_len)
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 768


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.a = nn.Parameter(torch.ones(config.d_model))
        self.q_weight = nn.Linear(config.d_model, config.d_model)
        self.k_weight = nn.Linear(config.d_model, config.d_model)
        self.v_weight = nn.Linear(config.d_model, config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head
        self.d_model = config.d_model

    def forward(self, q, k, v, mask: None):
        batch_size, seq_len, d_model = q.size()
        q = self.q_weight(q)
        k = self.k_weight(k)
        v = self.v_weight(v)
        q = q.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        k = k.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        v = v.view(batch_size, -1, self.n_head, self.d_model // self.n_head).transpose(
            1, 2
        )
        scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        if mask != None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1)
        y = scores @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        y = self.proj(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, config):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(config.d_model))
        self.b = nn.Parameter(torch.zeros(config.d_model))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        y = self.a * (x - mean) / (std + self.eps) + self.b
        return y


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm = LayerNorm(config)

    def forward(self, x, mask):
        x = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        x = self.layer_norm(x)
        return x


class BertInputEmbedding(nn.Module):

    def __init__(self, config):
        super(BertInputEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        sequence_length = x.size(1)
        pos = torch.arange(sequence_length, dtype=torch.long)
        # (sequence_length) -> (batch_size, sequence_length)
        pos = pos.unsqueeze(0).expand_as(x)
        x = self.token_embedding(x) + self.position_embedding(x)
        return x


class BERT(nn.Module):

    def __init__(self, config: BertConfig = None):
        super(BERT, self).__init__()
        assert config != None
        self.input_embedding = BertInputEmbedding(config)
        self.layers = nn.ModuleList(
            copy.deepcopy(Encoder(config)) for _ in range(config.n_layer)
        )

    def forward(self, x, mask):
        x = self.input_embedding(x)
        for encoder in self.layers:
            x = encoder(x, mask)
        return x


model = BERT(config=BertConfig())
