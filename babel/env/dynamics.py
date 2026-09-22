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
ALL_RESOURCE_TYPES: Final[tuple[str, ...]] = ("alpha", "beta", "gamma", "delta")
RESOURCE_TYPES: Final[tuple[str, str]] = ("alpha", "beta")
NUM_RESOURCES: Final[int] = len(RESOURCE_TYPES)
MAX_NEIGHBORHOOD_NODES: Final[int] = 8
MAX_VISIBLE_DEMANDS: Final[int] = 4
MAX_DEMANDS: Final[int] = 8
MESSAGE_HISTORY_LENGTH: Final[int] = 5
ACTION_DIM: Final[int] = 23
NOOP_ACTION: Final[int] = 22
BASE_OBSERVATION_DIM: Final[int] = 81
MESSAGE_HISTORY_OFFSET: Final[int] = 64
OBSERVATION_TAIL_DIM: Final[int] = BASE_OBSERVATION_DIM - MESSAGE_HISTORY_OFFSET
OBSERVATION_DIM: Final[int] = BASE_OBSERVATION_DIM
GLOBAL_STATE_DIM: Final[int] = 3 * NUM_RESOURCES + MAX_DEMANDS * (NUM_NODES + NUM_RESOURCES + 2)

MOVE_OFFSET: Final[int] = 0
PICKUP_OFFSET: Final[int] = 4
DROP_OFFSET: Final[int] = 6
TRANSFER_OFFSET: Final[int] = 8
COMMIT_OFFSET: Final[int] = 14


@dataclass(frozen=True, slots=True)
class ResourceLogisticsConfig:
    num_nodes: int = NUM_NODES
    num_agents: int = NUM_AGENTS
    num_resource_types: int = NUM_RESOURCES
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
    message_dim: int = 0
    message_history_length: int = MESSAGE_HISTORY_LENGTH
    private_demands: bool = False
    discovery_reward: float = 0.1
    role_asymmetry: bool = False
    semantic_channel: bool = False
    replay_path: Path | str | None = None
    # V22 additions
    num_depots: int = 3
    num_demand_nodes: int = 6
    num_transit_hubs: int = 3
    fuel_budget: int | None = None  # None = unlimited movement

    @property
    def resource_type_names(self) -> tuple[str, ...]:
        return ALL_RESOURCE_TYPES[: self.num_resource_types]

    # --- Dynamic dimension computation ---
    @property
    def move_offset(self) -> int:
        return 0

    @property
    def pickup_offset(self) -> int:
        return 4

    @property
    def drop_offset(self) -> int:
        return self.pickup_offset + self.num_resource_types

    @property
    def transfer_offset(self) -> int:
        return self.drop_offset + self.num_resource_types

    @property
    def commit_offset(self) -> int:
        return self.transfer_offset + (self.num_agents - 1) * self.num_resource_types

    @property
    def noop_action_id(self) -> int:
        return self.commit_offset + self.max_demands

    @property
    def action_dim(self) -> int:
        return self.noop_action_id + 1

    @property
    def base_observation_dim(self) -> int:
        """Observation dim WITHOUT message history."""
        pos = self.num_nodes  # position one-hot
        inv = self.num_resource_types  # own inventory
        neigh = MAX_NEIGHBORHOOD_NODES * (len(NODE_TYPES) + 1)  # neighborhood features
        depot = self.num_resource_types  # local depot inventory
        demands = MAX_VISIBLE_DEMANDS * (
            self.num_nodes + self.num_resource_types + 2
        )  # local demand features
        step = 1  # step counter
        commit = self.max_demands  # commitment features
        gps = MAX_NEIGHBORHOOD_NODES  # gps routing features
        fuel = 1 if self.fuel_budget is not None else 0  # fuel remaining
        return pos + inv + neigh + depot + demands + step + commit + gps + fuel

    @property
    def message_history_offset(self) -> int:
        """Index in the observation vector where message history begins.

        This equals the sum of: position one-hot + inventory + neighborhood
        + depot + demand features.
        The message history is inserted between demand features and the step counter.
        """
        pos = self.num_nodes
        inv = self.num_resource_types
        neigh = MAX_NEIGHBORHOOD_NODES * (len(NODE_TYPES) + 1)
        depot = self.num_resource_types
        demands = MAX_VISIBLE_DEMANDS * (self.num_nodes + self.num_resource_types + 2)
        return pos + inv + neigh + depot + demands

    @property
    def observation_dim(self) -> int:
        return (
            self.base_observation_dim
            + self.message_history_length * (self.num_agents - 1) * self.message_dim
        )

    @property
    def global_state_dim(self) -> int:
        return self.num_depots * self.num_resource_types + self.max_demands * (
            self.num_nodes + self.num_resource_types + 2
        )

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
            "resource_type": self.resource_type,
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
    fuel: int | None = None  # None = unlimited

    def inventory_total(self) -> int:
        return sum(self.inventory)

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "node_id": self.node_id,
            "inventory": {f"resource_{index}": count for index, count in enumerate(self.inventory)},
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
        self.noop_action = self.config.noop_action_id

        self.graph: ResourceGraph = generate_resource_graph(
            self.base_seed,
            num_nodes=self.config.num_nodes,
            num_depots=self.config.num_depots,
            num_demand_nodes=self.config.num_demand_nodes,
            num_transit_hubs=self.config.num_transit_hubs,
        )
        self.step_count = 0
        self.agent_states: dict[str, AgentState] = {}
        self.demands: list[DemandEvent] = []
        self.depot_inventory: dict[int, list[int]] = {}
        self.recorder: ReplayRecorder | None = None
        self.message_history: dict[str, NDArray[np.float32]] = {}
        self.revealed_demands: dict[str, set[int]] = {}
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

        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        self.graph = generate_resource_graph(
            episode_seed,
            num_nodes=self.config.num_nodes,
            num_depots=self.config.num_depots,
            num_demand_nodes=self.config.num_demand_nodes,
            num_transit_hubs=self.config.num_transit_hubs,
        )
        self.step_count = 0
        self.agents = list(self.possible_agents)
        self._finished = False

        start_nodes = self.rng.choice(
            self.config.num_nodes, size=self.config.num_agents, replace=False
        )
        self.agent_states = {
            agent: AgentState(
                node_id=int(start_nodes[index]),
                inventory=[0] * self.config.num_resource_types,
                fuel=self.config.fuel_budget,
            )
            for index, agent in enumerate(self.possible_agents)
        }
        self.depot_inventory = {
            depot: [
                int(
                    self.rng.integers(
                        self.config.depot_inventory_min, self.config.depot_inventory_max + 1
                    )
                )
                for _ in range(self.config.num_resource_types)
            ]
            for depot in self.graph.depots()
        }
        self.demands = self._sample_demands()
        self.revealed_demands = {agent: set() for agent in self.possible_agents}
        self.message_history = self._empty_message_history()
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
        actions: Mapping[str, int | Mapping[str, object]],
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
        action_payloads = {
            agent: actions.get(agent, self.noop_action) for agent in self.possible_agents
        }
        coerced_actions = {
            agent: self._coerce_action(action_payloads[agent]) for agent in self.possible_agents
        }

        team_reward = self._apply_actions(coerced_actions)
        self.step_count += 1
        team_reward += self._expire_demands()
        self._replenish_depots()
        self._update_message_history(action_payloads)

        # Private demand discovery: reveal demands at each agent's current node
        discovery_rewards: dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        if self.config.private_demands:
            for agent in self.possible_agents:
                agent_node = self.agent_states[agent].node_id
                for demand in self.demands:
                    if (
                        demand.status == "active"
                        and demand.node_id == agent_node
                        and demand.demand_id not in self.revealed_demands[agent]
                    ):
                        self.revealed_demands[agent].add(demand.demand_id)
                        discovery_rewards[agent] += self.config.discovery_reward

        terminations = {
            agent: self._all_demands_resolved() and self.step_count < self.config.episode_length
            for agent in self.possible_agents
        }
        truncations = {
            agent: self.step_count >= self.config.episode_length and not terminations[agent]
            for agent in self.possible_agents
        }
        rewards = {
            agent: float(team_reward) + discovery_rewards[agent] for agent in self.possible_agents
        }
        observations = self.observations()
        infos = self.infos()
        self._last_observations = observations

        self._record_step(observations_before, action_payloads, coerced_actions, rewards)

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

    def global_state_features(self) -> NDArray[np.float32]:
        cfg = self.config
        n_depots = cfg.num_depots
        n_res = cfg.num_resource_types
        n_nodes = cfg.num_nodes

        depot_features = np.zeros((n_depots, n_res), dtype=np.float32)
        for row, depot in enumerate(sorted(self.graph.depots())[:n_depots]):
            inventory = self.depot_inventory.get(depot, [0] * n_res)
            inventory_scale = max(float(cfg.depot_inventory_max), 1.0)
            depot_features[row] = np.clip(
                np.array(inventory, dtype=np.float32) / inventory_scale,
                0.0,
                1.0,
            )

        demand_width = n_nodes + n_res + 2
        demand_features = np.zeros((cfg.max_demands, demand_width), dtype=np.float32)
        active_demands = [demand for demand in self.demands if demand.status == "active"]
        active_demands.sort(key=lambda demand: (demand.deadline, demand.demand_id))
        for row, demand in enumerate(active_demands[: cfg.max_demands]):
            demand_features[row, demand.node_id] = 1.0
            demand_features[row, n_nodes + demand.resource_type] = 1.0
            remaining = max(demand.deadline - self.step_count, 0)
            demand_features[row, n_nodes + n_res] = min(
                float(remaining) / cfg.episode_length,
                1.0,
            )
            demand_features[row, n_nodes + n_res + 1] = float(demand.reward_value) / max(
                cfg.reward_values
            )

        features = np.concatenate(
            [depot_features.reshape(-1), demand_features.reshape(-1)],
            dtype=np.float32,
        )
        expected_dim = cfg.global_state_dim
        if features.shape != (expected_dim,):
            raise RuntimeError(f"Global state shape {features.shape} != {(expected_dim,)}")
        return features

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
                str(node_id): {f"resource_{index}": count for index, count in enumerate(inventory)}
                for node_id, inventory in self.depot_inventory.items()
            },
            "active_demands": [
                demand.to_dict() for demand in self.demands if demand.status == "active"
            ],
            "all_demands": [demand.to_dict() for demand in self.demands],
            "agent_positions": {agent: state.node_id for agent, state in self.agent_states.items()},
            "agent_inventories": {
                agent: {f"resource_{index}": count for index, count in enumerate(state.inventory)}
                for agent, state in self.agent_states.items()
            },
            "agent_commitments": {
                agent: state.commitment for agent, state in self.agent_states.items()
            },
            "message_history": {agent: history for agent, history in self.message_history.items()},
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
                    resource_type=int(self.rng.integers(0, self.config.num_resource_types)),
                    deadline=int(
                        self.rng.integers(self.config.deadline_min, self.config.deadline_max + 1)
                    ),
                    reward_value=int(self.rng.choice(self.config.reward_values)),
                )
            )
        return events

    def _apply_actions(self, actions: Mapping[str, int]) -> float:
        cfg = self.config
        reward = 0.0
        for agent, action in actions.items():
            mask = self._action_mask(agent)
            if action < 0 or action >= cfg.action_dim or not bool(mask[action]):
                action = self.noop_action
            if cfg.move_offset <= action < cfg.pickup_offset:
                self._move(agent, action - cfg.move_offset)
            elif cfg.pickup_offset <= action < cfg.drop_offset:
                self._pickup(agent, action - cfg.pickup_offset)
            elif cfg.drop_offset <= action < cfg.transfer_offset:
                reward += self._drop(agent, action - cfg.drop_offset)
            elif cfg.transfer_offset <= action < cfg.commit_offset:
                self._transfer(agent, action - cfg.transfer_offset)
            elif cfg.commit_offset <= action < cfg.noop_action_id:
                self._commit(agent, action - cfg.commit_offset)
        return reward

    def _move(self, agent: str, neighbor_slot: int) -> None:
        state = self.agent_states[agent]
        neighbors = self.graph.neighbors(state.node_id)
        if 0 <= neighbor_slot < len(neighbors):
            state.node_id = neighbors[neighbor_slot]
            if state.fuel is not None:
                state.fuel -= 1

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
        # if state.inventory[resource_type] <= 0:
        #     return 0.0
        # state.inventory[resource_type] -= 1
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
        resource_type = transfer_slot % self.config.num_resource_types
        target_index = transfer_slot // self.config.num_resource_types
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
            for resource_type in range(self.config.num_resource_types):
                if self.rng.random() < self.config.replenishment_rate:
                    inventory[resource_type] += 1

    def _all_demands_resolved(self) -> bool:
        return all(demand.status != "active" for demand in self.demands)

    def _action_mask(self, agent: str) -> NDArray[np.bool_]:
        cfg = self.config
        state = self.agent_states[agent]
        mask = np.zeros(cfg.action_dim, dtype=np.bool_)

        is_scout = cfg.role_asymmetry and agent in ["agent_0", "agent_1"]
        is_deliverer = cfg.role_asymmetry and agent in ["agent_2", "agent_3"]

        # Movement: only if agent has fuel remaining
        has_fuel = state.fuel is None or state.fuel > 0
        if has_fuel:
            for slot, _neighbor in enumerate(self.graph.neighbors(state.node_id)[:4]):
                mask[cfg.move_offset + slot] = True

        if not is_scout:
            depot_inventory = self.depot_inventory.get(state.node_id)
            if depot_inventory is not None and state.inventory_total() < cfg.inventory_capacity:
                for resource_type, count in enumerate(depot_inventory):
                    mask[cfg.pickup_offset + resource_type] = count > 0

            for resource_type, count in enumerate(state.inventory):
                mask[cfg.drop_offset + resource_type] = count > 0

            targets = [candidate for candidate in self.possible_agents if candidate != agent]
            for target_index, target in enumerate(targets):
                target_state = self.agent_states[target]
                if target_state.node_id != state.node_id:
                    continue
                if target_state.inventory_total() >= cfg.inventory_capacity:
                    continue
                for resource_type, count in enumerate(state.inventory):
                    if count > 0:
                        mask[
                            cfg.transfer_offset
                            + target_index * cfg.num_resource_types
                            + resource_type
                        ] = True

        if not is_deliverer:
            for demand in self.demands:
                if demand.demand_id < cfg.max_demands and demand.status == "active":
                    if cfg.private_demands and demand.demand_id not in self.revealed_demands.get(
                        agent, set()
                    ):
                        continue
                    mask[cfg.commit_offset + demand.demand_id] = True

        mask[self.noop_action] = True
        return mask

    def _observation_for(self, agent: str) -> NDArray[np.float32]:
        state = self.agent_states[agent]
        parts: list[NDArray[np.float32]] = []

        position = np.zeros(self.config.num_nodes, dtype=np.float32)
        position[state.node_id] = 1.0
        parts.append(position)

        parts.append(
            np.array(state.inventory, dtype=np.float32) / float(self.config.inventory_capacity)
        )
        parts.append(self._neighborhood_features(state.node_id))
        parts.append(self._local_depot_inventory_features(state.node_id))
        parts.append(self._local_demand_features(state.node_id, agent=agent))
        parts.append(self._message_history_features(agent))
        parts.append(np.array([self.step_count / self.config.episode_length], dtype=np.float32))
        parts.append(self._commitment_features(state.commitment))
        parts.append(self._gps_routing_features(agent))
        if self.config.fuel_budget is not None:
            fuel_frac = float(state.fuel or 0) / float(self.config.fuel_budget)
            parts.append(np.array([fuel_frac], dtype=np.float32))

        observation = np.concatenate(parts, dtype=np.float32)
        expected_shape = (self.config.observation_dim,)
        if observation.shape != expected_shape:
            raise RuntimeError(f"Observation shape {observation.shape} != {expected_shape}")
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

    def _gps_routing_features(self, agent: str) -> NDArray[np.float32]:
        features = np.zeros(MAX_NEIGHBORHOOD_NODES, dtype=np.float32)
        if self.config.message_dim == 0 or self.config.message_history_length == 0:
            return features

        latest_msgs = self.message_history[agent][-1]
        target_node = None
        for sender_idx in range(self.config.num_agents - 1):
            msg = latest_msgs[sender_idx]
            if np.sum(msg[: self.config.num_nodes]) > 0.5:
                target_node = int(np.argmax(msg[: self.config.num_nodes]))
                break

        if target_node is not None:
            node_id = self.agent_states[agent].node_id
            neighbors = self.graph.neighbors(node_id)
            for slot, neighbor in enumerate(neighbors[:MAX_NEIGHBORHOOD_NODES]):
                costs = self.graph.shortest_costs(neighbor)
                edge_cost = self.graph.edge_cost(node_id, neighbor)
                total_cost = edge_cost + costs[target_node]
                features[slot] = max(1.0 - (float(total_cost) / 20.0), 0.0)

        return features

    def _local_depot_inventory_features(self, node_id: int) -> NDArray[np.float32]:
        inventory = self.depot_inventory.get(node_id)
        if inventory is None:
            return np.zeros(self.config.num_resource_types, dtype=np.float32)
        return np.array(inventory, dtype=np.float32) / max(
            float(self.config.depot_inventory_max), 1.0
        )

    def _local_demand_features(self, node_id: int, agent: str | None = None) -> NDArray[np.float32]:
        is_scout = self.config.role_asymmetry and agent in ["agent_0", "agent_1"]
        is_deliverer = self.config.role_asymmetry and agent in ["agent_2", "agent_3"]

        if is_deliverer:
            visible_demands = []
        elif is_scout:
            visible_demands = [demand for demand in self.demands if demand.status == "active"]
        else:
            visible_nodes = {node_id, *self.graph.neighbors(node_id)}
            visible_demands = [
                demand
                for demand in self.demands
                if demand.status == "active" and demand.node_id in visible_nodes
            ]
            # Private demands: filter to only demands this agent has personally discovered
            if self.config.private_demands and agent is not None:
                revealed = self.revealed_demands.get(agent, set())
                visible_demands = [
                    demand for demand in visible_demands if demand.demand_id in revealed
                ]
        visible_demands.sort(key=lambda demand: (demand.deadline, demand.demand_id))
        demand_feature_width = self.config.num_nodes + self.config.num_resource_types + 2
        features = np.zeros((MAX_VISIBLE_DEMANDS, demand_feature_width), dtype=np.float32)
        for row, demand in enumerate(visible_demands[:MAX_VISIBLE_DEMANDS]):
            features[row, demand.node_id] = 1.0
            features[row, self.config.num_nodes + demand.resource_type] = 1.0
            remaining = max(demand.deadline - self.step_count, 0)
            features[row, self.config.num_nodes + self.config.num_resource_types] = min(
                float(remaining) / self.config.episode_length, 1.0
            )
            features[row, self.config.num_nodes + self.config.num_resource_types + 1] = float(
                demand.reward_value
            ) / max(self.config.reward_values)
        return features.reshape(-1)

    def _commitment_features(self, commitment: int | None) -> NDArray[np.float32]:
        features = np.zeros(self.config.max_demands, dtype=np.float32)
        if commitment is not None and 0 <= commitment < self.config.max_demands:
            features[commitment] = 1.0
        return features

    def _message_history_features(self, agent: str) -> NDArray[np.float32]:
        if self.config.message_dim == 0:
            return np.zeros(0, dtype=np.float32)
        return self.message_history[agent].reshape(-1).astype(np.float32)

    def _empty_message_history(self) -> dict[str, NDArray[np.float32]]:
        return {
            agent: np.zeros(
                (
                    self.config.message_history_length,
                    self.config.num_agents - 1,
                    self.config.message_dim,
                ),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def _update_message_history(self, actions: Mapping[str, object]) -> None:
        if self.config.message_dim == 0:
            return
        latest_by_receiver = {
            receiver: np.zeros(
                (self.config.num_agents - 1, self.config.message_dim),
                dtype=np.float32,
            )
            for receiver in self.possible_agents
        }
        for receiver in self.possible_agents:
            incoming = latest_by_receiver[receiver]
            senders = [sender for sender in self.possible_agents if sender != receiver]
            for sender_index, sender in enumerate(senders):
                if self.config.semantic_channel:
                    is_scout = self.config.role_asymmetry and sender in ["agent_0", "agent_1"]
                    if is_scout:
                        commitment = self.agent_states[sender].commitment
                        if commitment is not None:
                            demand = next(
                                (
                                    d
                                    for d in self.demands
                                    if d.demand_id == commitment and d.status == "active"
                                ),
                                None,
                            )
                            if demand is not None:
                                incoming[sender_index, demand.node_id] = 1.0
                                incoming[sender_index, 12 + demand.resource_type] = 1.0
                else:
                    embedding = self._message_embedding_from_action(actions.get(sender), receiver)
                    if embedding is not None:
                        incoming[sender_index] = embedding

        for receiver, latest in latest_by_receiver.items():
            if self.config.message_history_length == 0:
                continue
            history = self.message_history[receiver]
            history[:-1] = history[1:]
            history[-1] = latest

    def _message_embedding_from_action(
        self,
        action: object,
        receiver: str,
    ) -> NDArray[np.float32] | None:
        if not isinstance(action, Mapping):
            return None
        action_mapping = cast(Mapping[str, object], action)
        message_embeddings = action_mapping.get("message_embeddings")
        if not isinstance(message_embeddings, Mapping):
            return None
        embedding = cast(Mapping[str, object], message_embeddings).get(receiver)
        if embedding is None:
            return None
        array = np.asarray(embedding, dtype=np.float32)
        if array.shape != (self.config.message_dim,):
            raise ValueError(
                f"Message embedding for {receiver} has shape {array.shape}; "
                f"expected {(self.config.message_dim,)}"
            )
        return array

    def _record_step(
        self,
        observations: Mapping[str, NDArray[np.float32]],
        action_payloads: Mapping[str, object],
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
                    "message_sent_raw": self._message_raw_from_action(action_payloads.get(agent)),
                    "message_sent_perturbed": self._message_raw_from_action(
                        action_payloads.get(agent)
                    ),
                    "messages_received": self.message_history.get(
                        agent, np.zeros(0, dtype=np.float32)
                    ),
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

    def _message_raw_from_action(self, action: object) -> JsonLike:
        if not isinstance(action, Mapping):
            return ""
        action_mapping = cast(Mapping[str, object], action)
        message_raw = action_mapping.get("message_raw", "")
        return _json_value(message_raw)


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
