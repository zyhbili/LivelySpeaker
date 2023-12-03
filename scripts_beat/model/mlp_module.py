import torch
from torch import nn
import numpy as np

class LN_temporal(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_spatial(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class MLPblock(nn.Module):

    def __init__(self,seq_len , dim, dropout = 0.1, act = 'silu'):
        super().__init__()
        if act == 'relu':
            self.activation = nn.ReLU()
        if act == 'lrelu01':
            self.activation = nn.LeakyReLU(negative_slope=0.1)  
        if act == 'lrelu02':
            self.activation = nn.LeakyReLU(negative_slope=0.2)  
        if act == 'lrelu':
            self.activation = nn.LeakyReLU() 
        if act == 'silu':
            self.activation = nn.SiLU()
        self.block1 = nn.Sequential(
            LN_spatial(dim),
            nn.Conv1d(seq_len, seq_len, 1),
            self.activation,
        )
        self.block2 = nn.Sequential(
            LN_spatial(dim),
            nn.Linear(dim, dim),
            self.activation,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.block2[1].weight, gain=1e-8)
        nn.init.constant_(self.block2[1].bias, 0)

    def forward(self, x, emb = None):
        if emb is not None:
            x = x + emb
        x_ = self.block1(x)
        x = x + x_
        x_ = self.block2(x)
        x = x + x_
        return x

class TransMLP(nn.Module):
    def __init__(self, seq_len = 35,num_layers = 8, dim = 512, dropout = 0.1, act = 'silu'):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(seq_len, dim, dropout, act)
            for i in range(num_layers)])
        self.sequence_pos_encoder = PositionalEncoding(dim, dropout)
        self.embed_timestep  = TimestepEmbedder(dim, self.sequence_pos_encoder)

    def forward(self, x, t):
        emb = None
        emb = self.embed_timestep(t)
        for mlp in self.mlps:
            x = mlp(x, emb)
               
        return x

def build_mlps(seq_len = 35, num_layers = 8, hidden_dim = 512, dropout = 0.1, act = 'silu'):
    return TransMLP(
        seq_len = seq_len,
        num_layers=num_layers,
        dim=hidden_dim,     
        dropout=dropout, 
        act = act,
    )



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])



