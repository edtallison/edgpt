import torch
import torch.nn as nn

from config import GPTConfig
from attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_resid % config.n_heads == 0, "d_model must be divisible by n_heads"
        head_size = config.d_resid // config.n_heads

        self.heads = nn.ModuleList([
            AttentionHead(config, head_size) for _ in range(config.n_heads)
        ])
        self.heads_attn_mixing_fn = nn.Linear(config.d_resid, config.d_resid)

    def forward(self, resid_stream: torch.Tensor) -> torch.Tensor:
        # run each head independently, concat along the feature dim
        resid_stream = torch.cat([head_attn(resid_stream) for head_attn in self.heads], dim=-1)  # (B, T, d_model)
        resid_stream = self.heads_attn_mixing_fn(resid_stream)
        return resid_stream
