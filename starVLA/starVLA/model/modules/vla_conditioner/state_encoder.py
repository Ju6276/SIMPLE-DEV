import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    state -> continuous state features (not natural-language number strings).
    """

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 2:
            state = state.unsqueeze(1)
        return self.net(state)
