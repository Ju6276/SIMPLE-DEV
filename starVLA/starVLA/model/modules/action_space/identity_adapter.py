import torch
import torch.nn as nn


class IdentityActionAdapter(nn.Module):
    """
    Pass-through adapter: u := a (VLA-JEPA-DEV style raw-action interface).
    """

    def __init__(self, action_dim: int, control_dim: int, hidden_dim: int = 0):
        super().__init__()
        if action_dim != control_dim:
            raise ValueError(
                f"IdentityActionAdapter requires action_dim == control_dim, "
                f"got action_dim={action_dim}, control_dim={control_dim}."
            )
        self.action_dim = action_dim
        self.control_dim = control_dim
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def decode(self, controls: torch.Tensor) -> torch.Tensor:
        return controls

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return actions, actions
