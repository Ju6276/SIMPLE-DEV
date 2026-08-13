import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionAdapterBase(nn.Module):
    """
    Maps a raw action representation into the shared control latent u and
    decodes u back to the same raw action representation.
    """

    def __init__(
        self,
        action_dim: int,
        control_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = MLP(action_dim, hidden_dim, control_dim)
        self.decoder = MLP(control_dim, hidden_dim, action_dim)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        return self.encoder(actions)

    def decode(self, controls: torch.Tensor) -> torch.Tensor:
        return self.decoder(controls)

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        controls = self.encode(actions)
        reconstructed_actions = self.decode(controls)
        return controls, reconstructed_actions
