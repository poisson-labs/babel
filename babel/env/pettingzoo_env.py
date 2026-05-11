from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from babel.env.dynamics import (
    ACTION_DIM,
    OBSERVATION_DIM,
    ResourceLogisticsConfig,
    ResourceLogisticsEnv,
)


class ResourceLogisticsParallelEnv(ParallelEnv[str, NDArray[np.float32], int]):
    def __init__(
        self, config: ResourceLogisticsConfig | None = None, seed: int | None = None
    ) -> None:
        self.metadata: dict[str, Any] = {"name": "resource_logistics_v1", "render_modes": []}
        self.base_env = ResourceLogisticsEnv(config=config, seed=seed)
        self.possible_agents = list(self.base_env.possible_agents)
        self.agents = list(self.base_env.agents)
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self._action_space: spaces.Discrete[Any] = spaces.Discrete(ACTION_DIM)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, dict[str, Any]]]:
        del options
        observations, infos = self.base_env.reset(seed=seed)
        self.agents = list(self.base_env.agents)
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        observations, rewards, terminations, truncations, infos = self.base_env.step(actions)
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []
        else:
            self.agents = list(self.base_env.agents)
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> spaces.Box:
        self._check_agent(agent)
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Discrete[Any]:
        self._check_agent(agent)
        return self._action_space

    def close(self) -> None:
        self.base_env.close()

    def render(self) -> None:
        return None

    def _check_agent(self, agent: str) -> None:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent {agent!r}")


def parallel_env(
    config: ResourceLogisticsConfig | None = None,
    seed: int | None = None,
) -> ResourceLogisticsParallelEnv:
    return ResourceLogisticsParallelEnv(config=config, seed=seed)
