from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 5057
    d_resid: int = 256
    n_heads: int = 4
    n_blocks: int = 4
    context_len: int = 128
