from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from babel.channels.base import ChannelModule


@dataclass(frozen=True, slots=True)
class VQMessage:
    code: int
    embedding: Tensor


@dataclass(frozen=True, slots=True)
class VQBatch:
    slots: Tensor  # shape (batch_size, 1), contains the codebook indices
    embeddings: Tensor  # shape (batch_size, message_dim), the decoded message embeddings
    logprobs: Tensor  # shape (batch_size,), logprobs of the selection (zeros for VQ)
    entropy: Tensor  # shape (batch_size,), entropy (zeros for VQ)


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

        # Initialize ema_w with embeddings
        self.ema_w.data.copy_(self.embeddings.data)
        self.ema_cluster_size.data.fill_(1.0)

        self.decay = decay
        self.epsilon = epsilon
        self._current_loss = torch.tensor(0.0)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # inputs shape: (batch_size, embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.t())
        )

        # Encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1.0)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings).view(inputs.shape)

        # Use EMA to update the codebook during training
        if self.training:
            # Update EMA cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings, dim=0), alpha=1.0 - self.decay
            )

            # Laplace smoothing for cluster sizes
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size.data.copy_(
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            # Update EMA weights
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

            # Update embeddings
            self.embeddings.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        self._current_loss = loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.squeeze(-1)


class LatentVQChannel(nn.Module, ChannelModule[VQMessage]):
    def __init__(
        self,
        *,
        agent_intent_dim: int,
        observation_dim: int,
        message_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        codebook_size: int = 1024,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.agent_intent_dim = agent_intent_dim
        self.observation_dim = observation_dim
        self.message_dim = message_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        input_dim = agent_intent_dim + observation_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = nn.Linear(latent_dim, message_dim)
        self._last_loss = torch.tensor(0.0)

    def encode(self, agent_intent: Tensor, obs: Tensor, agent_id: int) -> VQMessage:
        del agent_id
        was_batched = agent_intent.ndim > 1
        if not was_batched:
            agent_intent = agent_intent.unsqueeze(0)
            obs = obs.unsqueeze(0)

        inputs = self.encoder(torch.cat([agent_intent, obs], dim=-1))
        quantized, _, indices = self.quantizer(inputs)

        return VQMessage(code=int(indices[0].item()), embedding=quantized[0])

    def decode(self, message: VQMessage, receiver_obs: Tensor) -> Tensor:
        del receiver_obs
        return self.decoder(message.embedding)

    def sample_batch(self, agent_intents: Tensor, observations: Tensor) -> VQBatch:
        inputs = self.encoder(torch.cat([agent_intents, observations], dim=-1))
        quantized, loss, indices = self.quantizer(inputs)
        self._last_loss = loss

        embeddings = self.decoder(quantized)
        batch_size = agent_intents.shape[0]
        zeros = torch.zeros(batch_size, dtype=embeddings.dtype, device=embeddings.device)

        return VQBatch(
            slots=indices.unsqueeze(-1),
            embeddings=embeddings,
            logprobs=zeros,
            entropy=zeros,
        )

    def deterministic_batch(self, agent_intents: Tensor, observations: Tensor) -> VQBatch:
        return self.sample_batch(agent_intents, observations)

    def evaluate_slots(
        self,
        agent_intents: Tensor,
        observations: Tensor,
        slots: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = agent_intents.shape[0]
        zeros = torch.zeros(batch_size, dtype=agent_intents.dtype, device=agent_intents.device)
        return zeros, zeros

    def capacity_bits(self) -> float:
        import math

        return math.log2(self.codebook_size)

    def get_loss(self) -> Tensor:
        return self._last_loss
