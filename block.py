import torch
import torch.nn as nn

from config import GPTConfig
from multi_head_attention import MultiHeadAttention
from mlp import MLP


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_resid)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_resid)
        self.mlp = MLP(config)

    def forward(self, resid_stream: torch.Tensor) -> torch.Tensor:
        resid_stream = resid_stream + self.attn(self.ln1(resid_stream))   # attention reads normed stream, writes raw residual
        resid_stream = resid_stream + self.mlp(self.ln2(resid_stream))    # MLP reads normed stream, writes raw residual
        return resid_stream
