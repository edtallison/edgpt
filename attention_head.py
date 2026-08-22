import torch
import torch.nn as nn
import math

from config import GPTConfig

class AttentionHead(nn.Module):
    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.q_proj_fn = nn.Linear(config.d_resid, head_size)
        self.k_proj_fn = nn.Linear(config.d_resid, head_size)
        self.v_proj_fn = nn.Linear(config.d_resid, head_size)

        causal_mask = torch.tril(torch.ones(config.context_len, config.context_len))
        # precompute once, not a learned parameter, so register as buffer
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, resid_stream: torch.Tensor) -> torch.Tensor:
        # resid_stream: (B, T, d_model)
        B, T, C = resid_stream.shape

        q = self.q_proj_fn(resid_stream)  # (B, T, head_size)
        k = self.k_proj_fn(resid_stream)  # (B, T, head_size)
        v = self.v_proj_fn(resid_stream)  # (B, T, head_size)

        # how much does each query token attend to each key token
        attn_scores = q @ k.transpose(-2, -1)  # (B, T, T)
        attn_scores = attn_scores / math.sqrt(self.head_size)  # scale

        # apply causal mask: token i can't attend to token j > i
        attn_scores = attn_scores.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)   # (B, T, T), rows sum to 1

        attn_from_this_head = attn @ v   # (B, T, head_size) — weighted sum of value vectors
        return attn_from_this_head