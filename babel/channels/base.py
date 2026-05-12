from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from torch import Tensor

Message = TypeVar("Message")


class Channel(Protocol[Message]):
    def encode(self, agent_intent: Tensor, obs: Tensor, agent_id: int) -> Message: ...

    def decode(self, message: Message, receiver_obs: Tensor) -> Tensor: ...

    def capacity_bits(self) -> float: ...


class ChannelModule(Generic[Message]):
    def encode(self, agent_intent: Tensor, obs: Tensor, agent_id: int) -> Message:
        raise NotImplementedError

    def decode(self, message: Message, receiver_obs: Tensor) -> Tensor:
        raise NotImplementedError

    def capacity_bits(self) -> float:
        raise NotImplementedError
