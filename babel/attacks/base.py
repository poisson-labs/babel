from typing import Any, Protocol

from babel.channels.base import ChannelModule
from babel.env.dynamics import ResourceLogisticsEnv


class CommsAttack(Protocol):
    def perturb(
        self,
        batch: Any,
        channel: ChannelModule,
        env: ResourceLogisticsEnv,
        step: int,
    ) -> Any: ...

    def report(self) -> dict[str, float]: ...
