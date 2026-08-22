import torch
import torch.nn as nn

from config import GPTConfig

class Embeddings(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_emb_fn = nn.Embedding(config.vocab_size, config.d_resid)
        self.pos_emb_fn = nn.Embedding(config.context_len, config.d_resid)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)  # (T,)

        tok_embedded = self.token_emb_fn(token_ids)  # (B, T, d_model)
        pos_embedded = self.pos_emb_fn(positions)  # (T, d_model), broadcasts over batch

        resid_stream = tok_embedded + pos_embedded  # (B, T, d_model)
        return resid_stream
