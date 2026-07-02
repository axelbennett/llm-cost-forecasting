import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4,
                 num_layers=2, horizon=7, n_quantiles=3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, horizon * n_quantiles)
        self.horizon = horizon
        self.n_quantiles = n_quantiles

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.output_head(x)
        return x.view(-1, self.horizon, self.n_quantiles)