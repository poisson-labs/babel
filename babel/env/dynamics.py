from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, ClassVar, Final, cast

import numpy as np
from numpy.typing import NDArray

from babel.env.graph import (
    NODE_TYPES,
    NUM_NODES,
    ResourceGraph,
    generate_resource_graph,
)
from babel.env.replay import JsonLike, ReplayRecorder

NUM_AGENTS: Final[int] = 4
RESOURCE_TYPES: Final[tuple[str, str]] = ("alpha", "beta")
NUM_RESOURCES: Final[int] = len(RESOURCE_TYPES)
MAX_NEIGHBORHOOD_NODES: Final[int] = 8
MAX_VISIBLE_DEMANDS: Final[int] = 4
MAX_DEMANDS: Final[int] = 8
ACTION_DIM: Final[int] = 23
NOOP_ACTION: Final[int] = 22
OBSERVATION_DIM: Final[int] = 73

MOVE_OFFSET: Final[int] = 0
PICKUP_OFFSET: Final[int] = 4
DROP_OFFSET: Final[int] = 6
TRANSFER_OFFSET: Final[int] = 8
COMMIT_OFFSET: Final[int] = 14


@dataclass(frozen=True, slots=True)
class ResourceLogisticsConfig:
    num_nodes: int = NUM_NODES
    num_agents: int = NUM_AGENTS
    episode_length: int = 50
    inventory_capacity: int = 2
    initial_demands_min: int = 5
    initial_demands_max: int = MAX_DEMANDS
    deadline_min: int = 15
    deadline_max: int = 35
    reward_values: tuple[int, ...] = (1, 2, 3)
    depot_inventory_min: int = 3
    depot_inventory_max: int = 5
    replenishment_rate: float = 0.1
    max_demands: int = MAX_DEMANDS
    replay_path: Path | str | None = None

    def to_dict(self) -> dict[str, JsonLike]:
        config = asdict(self)
        replay_path = config.get("replay_path")
        if replay_path is not None:
            config["replay_path"] = str(replay_path)
        return {key: _json_value(value) for key, value in config.items()}


@dataclass(slots=True)
class DemandEvent:
    demand_id: int
    node_id: int
    resource_type: int
    deadline: int
    reward_value: int
    status: str = "active"
    resolved_step: int | None = None
    resolved_by: str | None = None

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "demand_id": self.demand_id,
            "node_id": self.node_id,
            "resource_type": RESOURCE_TYPES[self.resource_type],
            "deadline": self.deadline,
            "reward_value": self.reward_value,
            "status": self.status,
            "resolved_step": self.resolved_step,
            "resolved_by": self.resolved_by,
        }


@dataclass(slots=True)
class AgentState:
    node_id: int
    inventory: list[int]
    commitment: int | None = None

    def inventory_total(self) -> int:
        return sum(self.inventory)

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "node_id": self.node_id,
            "inventory": {
                RESOURCE_TYPES[index]: count for index, count in enumerate(self.inventory)
            },
            "commitment": self.commitment,
        }


class ResourceLogisticsEnv:
    metadata: ClassVar[dict[str, str]] = {"name": "resource_logistics_v1"}

    def __init__(
        self, config: ResourceLogisticsConfig | None = None, seed: int | None = None
    ) -> None:
        self.config = config or ResourceLogisticsConfig()
        self.base_seed = int(seed or 0)
        self.rng = np.random.default_rng(self.base_seed)
        self.possible_agents = tuple(
            f"agent_{agent_id}" for agent_id in range(self.config.num_agents)
        )
        self.agents = list(self.possible_agents)
        self.noop_action = NOOP_ACTION

        self.graph: ResourceGraph = generate_resource_graph(self.base_seed)
        self.step_count = 0
        self.agent_states: dict[str, AgentState] = {}
        self.demands: list[DemandEvent] = []
        self.depot_inventory: dict[int, list[int]] = {}
        self.recorder: ReplayRecorder | None = None
        self._finished = False
        self._last_observations: dict[str, NDArray[np.float32]] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, dict[str, NDArray[np.bool_]]]]:
        if seed is not None:
            self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.graph = generate_resource_graph(self.base_seed)
        self.step_count = 0
        self.agents = list(self.possible_agents)
        self._finished = False

        start_nodes = self.rng.choice(NUM_NODES, size=self.config.num_agents, replace=False)
        self.agent_states = {
            agent: AgentState(node_id=int(start_nodes[index]), inventory=[0, 0])
            for index, agent in enumerate(self.possible_agents)
        }
        self.depot_inventory = {
            depot: [
                int(
                    self.rng.integers(
                        self.config.depot_inventory_min, self.config.depot_inventory_max + 1
                    )
                ),
                int(
                    self.rng.integers(
                        self.config.depot_inventory_min, self.config.depot_inventory_max + 1
                    )
                ),
            ]
            for depot in self.graph.depots()
        }
        self.demands = self._sample_demands()
        replay_path = Path(self.config.replay_path) if self.config.replay_path is not None else None
        self.recorder = ReplayRecorder(
            seed=self.base_seed, config=self.config.to_dict(), path=replay_path
        )

        observations = self.observations()
        infos = self.infos()
        self._last_observations = observations
        return observations, infos

    def step(
        self,
        actions: Mapping[str, int | Mapping[str, int]],
    ) -> tuple[
        dict[str, NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self._finished:
            raise RuntimeError(
                "Cannot call step() after the episode is finished; call reset() first."
            )

        observations_before = self._last_observations or self.observations()
        coerced_actions = {
            agent: self._coerce_action(actions.get(agent, self.noop_action))
            for agent in self.possible_agents
        }

        team_reward = self._apply_actions(coerced_actions)
        self.step_count += 1
        team_reward += self._expire_demands()
        self._replenish_depots()

        terminations = {
            agent: self._all_demands_resolved() and self.step_count < self.config.episode_length
            for agent in self.possible_agents
        }
        truncations = {
            agent: self.step_count >= self.config.episode_length and not terminations[agent]
            for agent in self.possible_agents
        }
        rewards = {agent: float(team_reward) for agent in self.possible_agents}
        observations = self.observations()
        infos = self.infos()
        self._last_observations = observations

        self._record_step(observations_before, coerced_actions, rewards)

        if all(terminations.values()) or all(truncations.values()):
            self._finished = True
            outcomes = self.outcomes()
            for info in infos.values():
                info["outcomes"] = outcomes
            if self.recorder is not None:
                self.recorder.finish(outcomes)

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        if self.recorder is not None and self._finished and self.recorder.path is not None:
            self.recorder.finish(self.outcomes())

    def observations(self) -> dict[str, NDArray[np.float32]]:
        return {agent: self._observation_for(agent) for agent in self.possible_agents}

    def infos(self) -> dict[str, dict[str, Any]]:
        masks = self.action_masks()
        return {agent: {"action_mask": masks[agent]} for agent in self.possible_agents}

    def action_masks(self) -> dict[str, NDArray[np.bool_]]:
        return {agent: self._action_mask(agent) for agent in self.possible_agents}

    def sample_valid_action(self, agent: str) -> int:
        mask = self._action_mask(agent)
        valid_actions = np.flatnonzero(mask)
        return int(self.rng.choice(valid_actions))

    def outcomes(self) -> dict[str, JsonLike]:
        satisfied = sum(1 for demand in self.demands if demand.status == "satisfied")
        missed = sum(1 for demand in self.demands if demand.status == "missed")
        total_reward = sum(
            demand.reward_value if demand.status == "satisfied" else -demand.reward_value
            for demand in self.demands
            if demand.status in {"satisfied", "missed"}
        )
        total = len(self.demands)
        return {
            "demands_satisfied": satisfied,
            "demands_missed": missed,
            "total_demands": total,
            "demand_satisfaction_rate": float(satisfied / total) if total > 0 else 0.0,
            "total_reward": float(total_reward),
        }

    def env_state(self) -> dict[str, JsonLike]:
        return {
            "step": self.step_count,
            "graph": self.graph.to_dict(),
            "node_inventory": {
                str(node_id): {
                    RESOURCE_TYPES[index]: count for index, count in enumerate(inventory)
                }
                for node_id, inventory in self.depot_inventory.items()
            },
            "active_demands": [
                demand.to_dict() for demand in self.demands if demand.status == "active"
            ],
            "all_demands": [demand.to_dict() for demand in self.demands],
            "agent_positions": {agent: state.node_id for agent, state in self.agent_states.items()},
            "agent_inventories": {
                agent: {RESOURCE_TYPES[index]: count for index, count in enumerate(state.inventory)}
                for agent, state in self.agent_states.items()
            },
            "agent_commitments": {
                agent: state.commitment for agent, state in self.agent_states.items()
            },
        }

    def _sample_demands(self) -> list[DemandEvent]:
        count = int(
            self.rng.integers(self.config.initial_demands_min, self.config.initial_demands_max + 1)
        )
        demand_nodes = np.array(self.graph.demand_nodes(), dtype=np.int64)
        sampled_nodes = self.rng.choice(demand_nodes, size=count, replace=True)
        events: list[DemandEvent] = []
        for demand_id, node_id in enumerate(sampled_nodes.tolist()):
            events.append(
                DemandEvent(
                    demand_id=demand_id,
                    node_id=int(node_id),
                    resource_type=int(self.rng.integers(0, NUM_RESOURCES)),
                    deadline=int(
                        self.rng.integers(self.config.deadline_min, self.config.deadline_max + 1)
                    ),
                    reward_value=int(self.rng.choice(self.config.reward_values)),
                )
            )
        return events

    def _apply_actions(self, actions: Mapping[str, int]) -> float:
        reward = 0.0
        for agent, action in actions.items():
            mask = self._action_mask(agent)
            if action < 0 or action >= ACTION_DIM or not bool(mask[action]):
                action = self.noop_action
            if MOVE_OFFSET <= action < PICKUP_OFFSET:
                self._move(agent, action - MOVE_OFFSET)
            elif PICKUP_OFFSET <= action < DROP_OFFSET:
                self._pickup(agent, action - PICKUP_OFFSET)
            elif DROP_OFFSET <= action < TRANSFER_OFFSET:
                reward += self._drop(agent, action - DROP_OFFSET)
            elif TRANSFER_OFFSET <= action < COMMIT_OFFSET:
                self._transfer(agent, action - TRANSFER_OFFSET)
            elif COMMIT_OFFSET <= action < NOOP_ACTION:
                self._commit(agent, action - COMMIT_OFFSET)
        return reward

    def _move(self, agent: str, neighbor_slot: int) -> None:
        state = self.agent_states[agent]
        neighbors = self.graph.neighbors(state.node_id)
        if 0 <= neighbor_slot < len(neighbors):
            state.node_id = neighbors[neighbor_slot]

    def _pickup(self, agent: str, resource_type: int) -> None:
        state = self.agent_states[agent]
        inventory = self.depot_inventory.get(state.node_id)
        if inventory is None:
            return
        if state.inventory_total() >= self.config.inventory_capacity:
            return
        if inventory[resource_type] <= 0:
            return
        inventory[resource_type] -= 1
        state.inventory[resource_type] += 1

    def _drop(self, agent: str, resource_type: int) -> float:
        state = self.agent_states[agent]
        if state.inventory[resource_type] <= 0:
            return 0.0
        state.inventory[resource_type] -= 1
        matching_demands = [
            demand
            for demand in self.demands
            if demand.status == "active"
            and demand.node_id == state.node_id
            and demand.resource_type == resource_type
            and self.step_count < demand.deadline
        ]
        if not matching_demands:
            return 0.0
        demand = min(matching_demands, key=lambda item: item.deadline)
        demand.status = "satisfied"
        demand.resolved_step = self.step_count
        demand.resolved_by = agent
        return float(demand.reward_value)

    def _transfer(self, agent: str, transfer_slot: int) -> None:
        resource_type = transfer_slot % NUM_RESOURCES
        target_index = transfer_slot // NUM_RESOURCES
        source_state = self.agent_states[agent]
        targets = [candidate for candidate in self.possible_agents if candidate != agent]
        if target_index >= len(targets):
            return
        target_state = self.agent_states[targets[target_index]]
        if source_state.node_id != target_state.node_id:
            return
        if source_state.inventory[resource_type] <= 0:
            return
        if target_state.inventory_total() >= self.config.inventory_capacity:
            return
        source_state.inventory[resource_type] -= 1
        target_state.inventory[resource_type] += 1

    def _commit(self, agent: str, demand_slot: int) -> None:
        if demand_slot >= len(self.demands):
            return
        demand = self.demands[demand_slot]
        if demand.status == "active":
            self.agent_states[agent].commitment = demand.demand_id

    def _expire_demands(self) -> float:
        penalty = 0.0
        for demand in self.demands:
            if demand.status == "active" and self.step_count >= demand.deadline:
                demand.status = "missed"
                demand.resolved_step = self.step_count
                penalty -= float(demand.reward_value)
        return penalty

    def _replenish_depots(self) -> None:
        for inventory in self.depot_inventory.values():
            for resource_type in range(NUM_RESOURCES):
                if self.rng.random() < self.config.replenishment_rate:
                    inventory[resource_type] += 1

    def _all_demands_resolved(self) -> bool:
        return all(demand.status != "active" for demand in self.demands)

    def _action_mask(self, agent: str) -> NDArray[np.bool_]:
        state = self.agent_states[agent]
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)

        for slot, _neighbor in enumerate(self.graph.neighbors(state.node_id)[:4]):
            mask[MOVE_OFFSET + slot] = True

        depot_inventory = self.depot_inventory.get(state.node_id)
        if depot_inventory is not None and state.inventory_total() < self.config.inventory_capacity:
            for resource_type, count in enumerate(depot_inventory):
                mask[PICKUP_OFFSET + resource_type] = count > 0

        for resource_type, count in enumerate(state.inventory):
            mask[DROP_OFFSET + resource_type] = count > 0

        targets = [candidate for candidate in self.possible_agents if candidate != agent]
        for target_index, target in enumerate(targets):
            target_state = self.agent_states[target]
            if target_state.node_id != state.node_id:
                continue
            if target_state.inventory_total() >= self.config.inventory_capacity:
                continue
            for resource_type, count in enumerate(state.inventory):
                if count > 0:
                    mask[TRANSFER_OFFSET + target_index * NUM_RESOURCES + resource_type] = True

        for demand in self.demands:
            if demand.demand_id < self.config.max_demands and demand.status == "active":
                mask[COMMIT_OFFSET + demand.demand_id] = True

        mask[self.noop_action] = True
        return mask

    def _observation_for(self, agent: str) -> NDArray[np.float32]:
        state = self.agent_states[agent]
        parts: list[NDArray[np.float32]] = []

        position = np.zeros(NUM_NODES, dtype=np.float32)
        position[state.node_id] = 1.0
        parts.append(position)

        parts.append(
            np.array(state.inventory, dtype=np.float32) / float(self.config.inventory_capacity)
        )
        parts.append(self._neighborhood_features(state.node_id))
        parts.append(self._local_depot_inventory_features(state.node_id))
        parts.append(self._local_demand_features(state.node_id))
        parts.append(np.array([self.step_count / self.config.episode_length], dtype=np.float32))
        parts.append(self._commitment_features(state.commitment))

        observation = np.concatenate(parts, dtype=np.float32)
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"Observation shape {observation.shape} != {(OBSERVATION_DIM,)}")
        return observation

    def _neighborhood_features(self, node_id: int) -> NDArray[np.float32]:
        hop_distances = self.graph.shortest_hops(node_id)
        travel_costs = self.graph.shortest_costs(node_id)
        visible_nodes = [
            candidate
            for candidate, hops in hop_distances.items()
            if candidate != node_id and hops <= 2
        ]
        visible_nodes.sort(
            key=lambda candidate: (hop_distances[candidate], travel_costs[candidate], candidate)
        )
        features = np.zeros((MAX_NEIGHBORHOOD_NODES, 4), dtype=np.float32)
        for row, visible_node in enumerate(visible_nodes[:MAX_NEIGHBORHOOD_NODES]):
            node_type_index = NODE_TYPES.index(self.graph.node_types[visible_node])
            features[row, node_type_index] = 1.0
            features[row, 3] = min(float(travel_costs[visible_node]) / 6.0, 1.0)
        return features.reshape(-1)

    def _local_depot_inventory_features(self, node_id: int) -> NDArray[np.float32]:
        inventory = self.depot_inventory.get(node_id)
        if inventory is None:
            return np.zeros(NUM_RESOURCES, dtype=np.float32)
        return np.array(inventory, dtype=np.float32) / max(
            float(self.config.depot_inventory_max), 1.0
        )

    def _local_demand_features(self, node_id: int) -> NDArray[np.float32]:
        visible_nodes = {node_id, *self.graph.neighbors(node_id)}
        visible_demands = [
            demand
            for demand in self.demands
            if demand.status == "active" and demand.node_id in visible_nodes
        ]
        visible_demands.sort(key=lambda demand: (demand.deadline, demand.demand_id))
        features = np.zeros((MAX_VISIBLE_DEMANDS, 4), dtype=np.float32)
        for row, demand in enumerate(visible_demands[:MAX_VISIBLE_DEMANDS]):
            features[row, demand.resource_type] = 1.0
            remaining = max(demand.deadline - self.step_count, 0)
            features[row, 2] = min(float(remaining) / self.config.episode_length, 1.0)
            features[row, 3] = float(demand.reward_value) / max(self.config.reward_values)
        return features.reshape(-1)

    def _commitment_features(self, commitment: int | None) -> NDArray[np.float32]:
        features = np.zeros(self.config.max_demands, dtype=np.float32)
        if commitment is not None and 0 <= commitment < self.config.max_demands:
            features[commitment] = 1.0
        return features

    def _record_step(
        self,
        observations: Mapping[str, NDArray[np.float32]],
        actions: Mapping[str, int],
        rewards: Mapping[str, float],
    ) -> None:
        if self.recorder is None:
            return
        per_agent: list[dict[str, Any]] = []
        for agent in self.possible_agents:
            per_agent.append(
                {
                    "agent_id": agent,
                    "obs": observations[agent],
                    "action": int(actions[agent]),
                    "message_sent_raw": "",
                    "message_sent_perturbed": "",
                    "messages_received": [],
                    "reward": float(rewards[agent]),
                }
            )
        self.recorder.record_step(
            step=self.step_count, env_state=self.env_state(), per_agent=per_agent
        )

    def _coerce_action(self, action: object) -> int:
        if isinstance(action, Mapping):
            action_mapping = cast(Mapping[str, object], action)
            movement = action_mapping.get("movement", self.noop_action)
            if isinstance(movement, Integral):
                return int(movement)
            return self.noop_action
        if isinstance(action, Integral):
            return int(action)
        return self.noop_action


def _json_value(value: Any) -> JsonLike:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple):
        return [_json_value(item) for item in cast(tuple[object, ...], value)]
    if isinstance(value, list):
        return [_json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item) for key, item in cast(dict[object, object], value).items()
        }
    raise TypeError(f"Unsupported config value type: {type(value)!r}")
