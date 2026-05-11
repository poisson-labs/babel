from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from babel.env.dynamics import OBSERVATION_DIM
from babel.networks.encoders import ObservationEncoder

_torch = cast(Any, torch)


class IPPOCritic(nn.Module):
    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = ObservationEncoder(observation_dim=observation_dim, hidden_dim=hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: object) -> torch.Tensor:
        return cast(
            torch.Tensor,
            _torch.zeros((batch_size, self.hidden_dim), dtype=_torch.float32, device=device),
        )

    def forward_step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observations)
        next_hidden = self.gru(encoded, hidden)
        values = self.value_head(next_hidden).squeeze(-1)
        return values, next_hidden
