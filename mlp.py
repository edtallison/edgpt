import torch
import torch.nn as nn

from config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1_expand = nn.Linear(config.d_resid, 4 * config.d_resid)
        self.activation = nn.GELU()
        self.fc2_contract = nn.Linear(4 * config.d_resid, config.d_resid)
        self.dropout = nn.Dropout(0.1)

    def forward(self, resid_stream: torch.Tensor) -> torch.Tensor:
        # resid_stream: (B, T, d_model) — every position processed independently
        resid_stream = self.fc1_expand(resid_stream)  # (B, T, 4*d_model)
        resid_stream = self.activation(resid_stream)  # nonlinearity — this is where the "thinking" happens
        resid_stream = self.fc2_contract(resid_stream)  # (B, T, d_model)
        return self.dropout(resid_stream)
