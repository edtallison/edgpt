import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 5057
    residual_stream_dim: int = 256
    n_heads: int = 4
    n_blocks: int = 4
    context_len: int = 128
    dropout: float = 0.1

