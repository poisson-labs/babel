from __future__ import annotations

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from dataclasses import dataclass
from typing import Any, Final, cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from babel.channels.base import ChannelModule

CLAIM_TYPES: Final[tuple[str, ...]] = (
    "at_depot",
    "going_to",
    "committed_to",
    "route_blocked",
    "have_resource",
    "need_resource",
    "ack",
    "null",
)
SYMBOLIC_CAPACITY_BITS: Final[int] = 8
SYMBOLIC_SLOT_SIZES: Final[tuple[int, int, int]] = (8, 16, 2)
_torch = cast(Any, torch)


@dataclass(frozen=True, slots=True)
class SymbolicMessage:
    claim_type: int
    node_id: int
    payload: int

    def __post_init__(self) -> None:
        if not 0 <= self.claim_type < SYMBOLIC_SLOT_SIZES[0]:
            raise ValueError(f"claim_type must be in [0, 7], got {self.claim_type}")
        if not 0 <= self.node_id < SYMBOLIC_SLOT_SIZES[1]:
            raise ValueError(f"node_id must be in [0, 15], got {self.node_id}")
        if not 0 <= self.payload < SYMBOLIC_SLOT_SIZES[2]:
            raise ValueError(f"payload must be in [0, 1], got {self.payload}")

    @property
    def code(self) -> int:
        return self.claim_type * 32 + self.node_id * 2 + self.payload

    def to_slots(self) -> tuple[int, int, int]:
        return (self.claim_type, self.node_id, self.payload)

    def to_bits(self) -> tuple[int, ...]:
        return (
            *_bits(self.claim_type, 3),
            *_bits(self.node_id, 4),
            *_bits(self.payload, 1),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "claim_type": self.claim_type,
            "claim_name": CLAIM_TYPES[self.claim_type],
            "node_id": self.node_id,
            "payload": self.payload,
            "code": self.code,
        }

    @classmethod
    def from_bits(cls, bits: tuple[int, ...] | list[int]) -> SymbolicMessage:
        if len(bits) != SYMBOLIC_CAPACITY_BITS:
            raise ValueError(f"Expected 8 bits, got {len(bits)}")
        if any(bit not in {0, 1} for bit in bits):
            raise ValueError("Symbolic message bits must be binary")
        claim_type = _from_bits(bits[0:3])
        node_id = _from_bits(bits[3:7])
        payload = _from_bits(bits[7:8])
        return cls(claim_type=claim_type, node_id=node_id, payload=payload)

    @classmethod
    def from_slots(cls, slots: Tensor) -> SymbolicMessage:
        cpu_slots = slots.detach().to("cpu").long().tolist()
        if len(cpu_slots) != 3:
            raise ValueError("Expected a 3-slot symbolic tensor")
        return cls(
            claim_type=int(cpu_slots[0]),
            node_id=int(cpu_slots[1]),
            payload=int(cpu_slots[2]),
        )


@dataclass(frozen=True, slots=True)
class SymbolicBatch:
    slots: Tensor
    embeddings: Tensor
    logprobs: Tensor
    entropy: Tensor


class SymbolicChannel(nn.Module, ChannelModule[SymbolicMessage]):
    def __init__(
        self,
        *,
        agent_intent_dim: int,
        observation_dim: int,
        message_dim: int = 16,
        hidden_dim: int = 128,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.2,
        anneal_steps: int = 1_000_000,
        slot_sizes: tuple[int, ...] = SYMBOLIC_SLOT_SIZES,
    ) -> None:
        super().__init__()
        self.agent_intent_dim = agent_intent_dim
        self.observation_dim = observation_dim
        self.message_dim = message_dim
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_steps = max(anneal_steps, 1)
        self.temperature = initial_temperature
        self.slot_sizes = slot_sizes

        # Compute capacity bits from slot sizes: sum of log2(each slot size)
        import math

        self._capacity_bits = sum(math.log2(s) for s in slot_sizes)
        # Compute multipliers for code packing (big-endian)
        self._slot_multipliers: list[int] = []
        mult = 1
        for s in reversed(slot_sizes):
            self._slot_multipliers.insert(0, mult)
            mult *= s
        self._total_codes = mult

        input_dim = agent_intent_dim + observation_dim
        self.sender = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.slot_heads = nn.ModuleList([nn.Linear(hidden_dim, size) for size in slot_sizes])
        self.decoder = nn.Embedding(self._total_codes, message_dim)

    def encode(self, agent_intent: Tensor, obs: Tensor, agent_id: int) -> SymbolicMessage:
        del agent_id
        was_batched = agent_intent.ndim > 1
        if not was_batched:
            agent_intent = agent_intent.unsqueeze(0)
            obs = obs.unsqueeze(0)
        logits = self._slot_logits(agent_intent, obs)
        slots = []
        for slot_logits in logits:
            if self.training:
                one_hot = F.gumbel_softmax(
                    slot_logits,
                    tau=self.temperature,
                    hard=True,
                    dim=-1,
                )
                slots.append(one_hot.argmax(dim=-1))
            else:
                slots.append(slot_logits.argmax(dim=-1))
        stacked_slots = _torch.stack(slots, dim=-1)
        return SymbolicMessage.from_slots(stacked_slots[0])

    def decode(self, message: SymbolicMessage, receiver_obs: Tensor) -> Tensor:
        del receiver_obs
        code = _torch.as_tensor(message.code, dtype=_torch.long, device=self.decoder.weight.device)
        return cast(Tensor, self.decoder(code))

    def decode_slots(self, slots: Tensor) -> Tensor:
        codes = self.slots_to_codes(slots)
        return cast(Tensor, self.decoder(codes))

    def sample_batch(self, agent_intents: Tensor, observations: Tensor) -> SymbolicBatch:
        logits = self._slot_logits(agent_intents, observations)
        slot_values: list[Tensor] = []
        logprobs: list[Tensor] = []
        entropies: list[Tensor] = []
        for slot_logits in logits:
            relaxed = F.gumbel_softmax(slot_logits, tau=self.temperature, hard=True, dim=-1)
            sampled_slots = relaxed.argmax(dim=-1)
            distribution = Categorical(logits=slot_logits)
            slot_values.append(sampled_slots)
            logprobs.append(distribution.log_prob(sampled_slots))
            entropies.append(distribution.entropy())

        slots = _torch.stack(slot_values, dim=-1)
        embeddings = self.decode_slots(slots)
        return SymbolicBatch(
            slots=slots,
            embeddings=embeddings,
            logprobs=_torch.stack(logprobs, dim=0).sum(dim=0),
            entropy=_torch.stack(entropies, dim=0).sum(dim=0),
        )

    def deterministic_batch(self, agent_intents: Tensor, observations: Tensor) -> SymbolicBatch:
        logits = self._slot_logits(agent_intents, observations)
        slot_values = [slot_logits.argmax(dim=-1) for slot_logits in logits]
        slots = _torch.stack(slot_values, dim=-1)
        embeddings = self.decode_slots(slots)
        batch_size = slots.shape[0]
        zeros = _torch.zeros(batch_size, dtype=embeddings.dtype, device=embeddings.device)
        return SymbolicBatch(slots=slots, embeddings=embeddings, logprobs=zeros, entropy=zeros)

    def evaluate_slots(
        self,
        agent_intents: Tensor,
        observations: Tensor,
        slots: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self._slot_logits(agent_intents, observations)
        logprobs: list[Tensor] = []
        entropies: list[Tensor] = []
        for slot_index, slot_logits in enumerate(logits):
            distribution = Categorical(logits=slot_logits)
            slot_values = slots[:, slot_index].long()
            logprobs.append(distribution.log_prob(slot_values))
            entropies.append(distribution.entropy())
        return (
            _torch.stack(logprobs, dim=0).sum(dim=0),
            _torch.stack(entropies, dim=0).sum(dim=0),
        )

    def capacity_bits(self) -> float:
        return self._capacity_bits

    def anneal_temperature(self, step: int) -> float:
        fraction = min(max(step / self.anneal_steps, 0.0), 1.0)
        self.temperature = self.initial_temperature + fraction * (
            self.min_temperature - self.initial_temperature
        )
        return self.temperature

    def slots_to_messages(self, slots: Tensor) -> list[SymbolicMessage]:
        return [SymbolicMessage.from_slots(row) for row in slots]

    def slots_to_codes(self, slots: Tensor) -> Tensor:
        slots = slots.long()
        code = _torch.zeros(slots.shape[0], dtype=_torch.long, device=slots.device)
        for i, mult in enumerate(self._slot_multipliers):
            code = code + slots[:, i] * mult
        return code

    def _slot_logits(
        self,
        agent_intents: Tensor,
        observations: Tensor,
    ) -> tuple[Tensor, ...]:
        features = self.sender(_torch.cat([agent_intents, observations], dim=-1))
        return tuple(head(features) for head in self.slot_heads)


def _bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in reversed(range(width)))


def _from_bits(bits: tuple[int, ...] | list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value
