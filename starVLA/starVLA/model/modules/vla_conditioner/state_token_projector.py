import torch
import torch.nn as nn


class StateTokenProjector(nn.Module):
    """
    Project continuous state features into Qwen-compatible state tokens.
    """

    def __init__(
        self,
        input_dim: int,
        qwen_hidden_dim: int,
        num_tokens: int = 4,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        hidden = hidden_dim if hidden_dim is not None else qwen_hidden_dim
        self.pre = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
        )
        self.token_heads = nn.ModuleList(
            [nn.Linear(hidden, qwen_hidden_dim) for _ in range(num_tokens)]
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        if state_features.dim() == 3:
            pooled = state_features.mean(dim=1)
        else:
            pooled = state_features
        hidden = self.pre(pooled)
        tokens = [head(hidden) for head in self.token_heads]
        return torch.stack(tokens, dim=1)
