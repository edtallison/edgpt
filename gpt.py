import torch
import torch.nn as nn
import math

from config import GPTConfig
from embeddings import Embeddings
from block import Block


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)])
        self.ln_final = nn.LayerNorm(config.d_resid)
        self.lm_head = nn.Linear(config.d_resid, config.vocab_size, bias=False)

        # weight tying: share embedding and unembedding weights
        self.lm_head.weight = self.embeddings.token_emb_fn.weight

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor = None):
        # token_ids: (B, T)
        resid_stream = self.embeddings(token_ids)          # (B, T, d_model)

        for block in self.blocks:
            resid_stream = block(resid_stream)                   # (B, T, d_model)

        resid_stream = self.ln_final(resid_stream)               # final norm before reading out
        logits = self.lm_head(resid_stream)           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten batch and time dims for cross_entropy
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
        return logits, loss
