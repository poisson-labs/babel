from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from babel.algorithms.mappo import MAPPOActor, SymbolicChannelConfig
from babel.channels.symbolic import CLAIM_TYPES, SYMBOLIC_CAPACITY_BITS, SymbolicChannel
from babel.env.dynamics import (
    NUM_AGENTS,
    NUM_RESOURCES,
    ResourceLogisticsConfig,
    ResourceLogisticsEnv,
)
from babel.env.graph import DEPOT, TRANSIT
from babel.networks.actor import IPPOActor

DEVICE = "cpu"
SYMBOLIC_EPISODES_PER_SEED = 200
IPPO_EPISODES_PER_SEED = 200
SOLO_ENV_EPISODES = 200
SYMBOLIC_EVAL_SEED = 200_000
IPPO_EVAL_SEED = 100_000
SOLO_ENV_SEED = 300_000
NULL_CLAIM = CLAIM_TYPES.index("null")
DEFAULT_MESSAGE = (NULL_CLAIM, 0, 0)


@dataclass(slots=True)
class CargoItem:
    resource_type: int
    source_depot: int | None
    pickup_agent: str
    current_agent: str
    pickup_step: int
    transferred: bool = False
    depot_colocated: bool = False
    transit_rendezvous: bool = False


@dataclass(slots=True)
class DeliveryRecord:
    agent: str
    demand_id: int
    resource_type: int
    step: int
    source_depot: int | None
    pickup_agent: str | None
    transferred: bool
    depot_colocated: bool
    transit_rendezvous: bool
    provenance_known: bool

    @property
    def solo_success(self) -> bool:
        return (
            self.provenance_known
            and self.source_depot is not None
            and self.pickup_agent == self.agent
            and not self.transferred
            and not self.depot_colocated
        )

    @property
    def coordinated(self) -> bool:
        return self.transferred or self.transit_rendezvous


class DiagnosticEnv(ResourceLogisticsEnv):
    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        observations, infos = super().reset(seed=seed)
        self.cargo: dict[str, dict[int, list[CargoItem]]] = {
            agent: {resource_type: [] for resource_type in range(NUM_RESOURCES)}
            for agent in self.possible_agents
        }
        self.delivery_records: list[DeliveryRecord] = []
        return observations, infos

    def step(self, actions: dict[str, int | dict[str, object]]):  # type: ignore[no-untyped-def]
        self._mark_colocations_for_active_cargo()
        return super().step(actions)

    def _pickup(self, agent: str, resource_type: int) -> None:
        state = self.agent_states[agent]
        node_id = state.node_id
        before = state.inventory[resource_type]
        super()._pickup(agent, resource_type)
        if state.inventory[resource_type] <= before:
            return
        self.cargo[agent][resource_type].append(
            CargoItem(
                resource_type=resource_type,
                source_depot=node_id if self.graph.node_types[node_id] == DEPOT else None,
                pickup_agent=agent,
                current_agent=agent,
                pickup_step=self.step_count,
                depot_colocated=self._has_colocated_agent(agent, node_id)
                if self.graph.node_types[node_id] == DEPOT
                else False,
            )
        )

    def _transfer(self, agent: str, transfer_slot: int) -> None:
        resource_type = transfer_slot % NUM_RESOURCES
        target_index = transfer_slot // NUM_RESOURCES
        targets = [candidate for candidate in self.possible_agents if candidate != agent]
        if target_index >= len(targets):
            super()._transfer(agent, transfer_slot)
            return
        target = targets[target_index]
        before_source = self.agent_states[agent].inventory[resource_type]
        before_target = self.agent_states[target].inventory[resource_type]
        super()._transfer(agent, transfer_slot)
        source_decreased = self.agent_states[agent].inventory[resource_type] < before_source
        target_increased = self.agent_states[target].inventory[resource_type] > before_target
        if not source_decreased or not target_increased:
            return
        item = self._pop_cargo(agent, resource_type)
        if item is None:
            return
        item.transferred = True
        item.current_agent = target
        self.cargo[target][resource_type].append(item)

    def _drop(self, agent: str, resource_type: int) -> float:
        before_inventory = self.agent_states[agent].inventory[resource_type]
        before_status = {demand.demand_id: demand.status for demand in self.demands}
        reward = super()._drop(agent, resource_type)
        after_inventory = self.agent_states[agent].inventory[resource_type]
        if after_inventory >= before_inventory:
            return reward

        item = self._pop_cargo(agent, resource_type)
        delivered = [
            demand
            for demand in self.demands
            if before_status[demand.demand_id] == "active" and demand.status == "satisfied"
        ]
        if delivered:
            demand = delivered[0]
            self.delivery_records.append(
                DeliveryRecord(
                    agent=agent,
                    demand_id=demand.demand_id,
                    resource_type=resource_type,
                    step=self.step_count,
                    source_depot=item.source_depot if item is not None else None,
                    pickup_agent=item.pickup_agent if item is not None else None,
                    transferred=item.transferred if item is not None else False,
                    depot_colocated=item.depot_colocated if item is not None else False,
                    transit_rendezvous=item.transit_rendezvous if item is not None else False,
                    provenance_known=item is not None,
                )
            )
        return reward

    def _pop_cargo(self, agent: str, resource_type: int) -> CargoItem | None:
        items = self.cargo[agent][resource_type]
        if not items:
            return None
        return items.pop(0)

    def _mark_colocations_for_active_cargo(self) -> None:
        positions: dict[int, list[str]] = {}
        for agent, state in self.agent_states.items():
            positions.setdefault(state.node_id, []).append(agent)
        for node_id, agents in positions.items():
            if len(agents) <= 1:
                continue
            node_type = self.graph.node_types[node_id]
            if node_type not in {DEPOT, TRANSIT}:
                continue
            for agent in agents:
                for by_resource in self.cargo[agent].values():
                    for item in by_resource:
                        if node_type == DEPOT:
                            item.depot_colocated = True
                        elif node_type == TRANSIT:
                            item.transit_rendezvous = True

    def _has_colocated_agent(self, agent: str, node_id: int) -> bool:
        return any(
            other_agent != agent and state.node_id == node_id
            for other_agent, state in self.agent_states.items()
        )


def main() -> None:
    torch.set_num_threads(1)
    output = run_diagnostics()
    artifact_path = Path("artifacts/phase2_gap_diagnostics.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(output, indent=2) + "\n")

    report_path = Path("docs/diagnostics/phase2_gap_analysis.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(output) + "\n")
    print(f"Wrote {report_path}")
    print(f"Wrote {artifact_path}")
    print(f"Recommendation: {output['recommendation']}")


def run_diagnostics() -> dict[str, Any]:
    symbolic_runs = []
    for seed_index in range(2):
        symbolic_runs.append(
            evaluate_symbolic(
                checkpoint_path=Path(f"artifacts/checkpoints/mappo_symbolic_seed_{seed_index}.pt"),
                episodes=SYMBOLIC_EPISODES_PER_SEED,
                seed=SYMBOLIC_EVAL_SEED + seed_index * 500,
            )
        )
    ippo_runs = []
    for seed_index in range(2):
        ippo_runs.append(
            evaluate_ippo(
                checkpoint_path=Path(f"artifacts/checkpoints/ippo_seed_{seed_index}.pt"),
                episodes=IPPO_EPISODES_PER_SEED,
                seed=IPPO_EVAL_SEED + seed_index * 500,
            )
        )
    solo = analyze_solo_completability(
        episodes=SOLO_ENV_EPISODES,
        seed=SOLO_ENV_SEED,
    )

    symbolic_aggregate = aggregate_policy_runs(symbolic_runs)
    ippo_aggregate = aggregate_policy_runs(ippo_runs)
    channel_aggregate = aggregate_channel_runs(symbolic_runs)
    recommendation = choose_recommendation(
        empirical_bits=channel_aggregate["slot_entropy_total_bits"],
        solo_completable_fraction=solo["solo_completable_fraction"],
        ippo_solo_success_fraction=ippo_aggregate["solo_success_fraction"],
        ippo_coordination_fraction=ippo_aggregate["coordination_fraction"],
        symbolic_coordination_fraction=symbolic_aggregate["coordination_fraction"],
    )
    return {
        "metadata": {
            "symbolic_episodes_per_seed": SYMBOLIC_EPISODES_PER_SEED,
            "ippo_episodes_per_seed": IPPO_EPISODES_PER_SEED,
            "solo_env_episodes": SOLO_ENV_EPISODES,
            "symbolic_eval_seed": SYMBOLIC_EVAL_SEED,
            "ippo_eval_seed": IPPO_EVAL_SEED,
            "solo_env_seed": SOLO_ENV_SEED,
            "gate2_gap_vs_ippo_point_pp": 2.9,
            "gate2_gap_vs_ippo_upper_ci_pp": 1.6,
        },
        "channel_utilization": {
            "by_seed": [run["channel"] for run in symbolic_runs],
            "aggregate": channel_aggregate,
        },
        "solo_completability": solo,
        "ippo_behavior": {
            "by_seed": [run["behavior"] for run in ippo_runs],
            "aggregate": ippo_aggregate,
        },
        "symbolic_behavior": {
            "by_seed": [run["behavior"] for run in symbolic_runs],
            "aggregate": symbolic_aggregate,
        },
        "recommendation": recommendation,
    }


def evaluate_symbolic(
    *,
    checkpoint_path: Path,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    channel_config = SymbolicChannelConfig(**checkpoint["channel_config"])
    env_config = ResourceLogisticsConfig(
        message_dim=channel_config.message_dim,
        message_history_length=5,
        deadline_min=12,
        deadline_max=20,
        depot_inventory_min=3,
        depot_inventory_max=5,
    )
    actor = MAPPOActor(
        observation_dim=env_config.observation_dim,
        message_dim=channel_config.message_dim,
    ).to(DEVICE)
    channel = SymbolicChannel(
        agent_intent_dim=actor.hidden_dim,
        observation_dim=env_config.observation_dim,
        message_dim=channel_config.message_dim,
        hidden_dim=channel_config.hidden_dim,
        initial_temperature=channel_config.initial_temperature,
        min_temperature=channel_config.min_temperature,
        anneal_steps=channel_config.anneal_steps,
    ).to(DEVICE)
    actor.load_state_dict(checkpoint["actor"])
    channel.load_state_dict(checkpoint["channel"])
    actor.eval()
    channel.eval()

    all_slots: list[tuple[int, int, int]] = []
    deliveries: list[DeliveryRecord] = []
    satisfaction_rates: list[float] = []
    for episode in range(episodes):
        env = DiagnosticEnv(config=env_config, seed=seed + episode)
        observations, infos = env.reset(seed=seed + episode)
        hidden = actor.initial_hidden(NUM_AGENTS, DEVICE)
        done = False
        while not done:
            obs_tensor = stack_observations(observations, env)
            mask_tensor = stack_action_masks(infos, env)
            with torch.no_grad():
                logits, hidden, agent_intents = actor.forward_step(obs_tensor, hidden, mask_tensor)
                movement_actions = torch.argmax(logits, dim=-1)
                message_batch = channel.deterministic_batch(agent_intents, obs_tensor)
            slots = message_batch.slots.cpu().numpy().astype(int)
            all_slots.extend(tuple(int(value) for value in row) for row in slots.tolist())
            action_dict = symbolic_actions(
                movement_actions.cpu().numpy(),
                message_batch.embeddings.cpu().numpy(),
                slots,
                env,
            )
            observations, _rewards, terminations, truncations, infos = env.step(action_dict)
            done = all(terminations.values()) or all(truncations.values())
        satisfaction_rates.append(float(env.outcomes()["demand_satisfaction_rate"]))
        deliveries.extend(env.delivery_records)
        env.close()

    return {
        "checkpoint": str(checkpoint_path),
        "channel": summarize_channel(all_slots, checkpoint_path.name),
        "behavior": summarize_deliveries(deliveries, satisfaction_rates, checkpoint_path.name),
    }


def evaluate_ippo(
    *,
    checkpoint_path: Path,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env_config = ResourceLogisticsConfig(
        deadline_min=12,
        deadline_max=20,
        depot_inventory_min=3,
        depot_inventory_max=5,
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    actor = IPPOActor(observation_dim=env_config.observation_dim).to(DEVICE)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    deliveries: list[DeliveryRecord] = []
    satisfaction_rates: list[float] = []
    for episode in range(episodes):
        env = DiagnosticEnv(config=env_config, seed=seed + episode)
        observations, infos = env.reset(seed=seed + episode)
        hidden = actor.initial_hidden(NUM_AGENTS, DEVICE)
        done = False
        while not done:
            obs_tensor = stack_observations(observations, env)
            mask_tensor = stack_action_masks(infos, env)
            with torch.no_grad():
                logits, hidden = actor.forward_step(obs_tensor, hidden, mask_tensor)
                actions = torch.argmax(logits, dim=-1)
            action_dict = {
                agent: int(actions[index].item()) for index, agent in enumerate(env.possible_agents)
            }
            observations, _rewards, terminations, truncations, infos = env.step(action_dict)
            done = all(terminations.values()) or all(truncations.values())
        satisfaction_rates.append(float(env.outcomes()["demand_satisfaction_rate"]))
        deliveries.extend(env.delivery_records)
        env.close()

    return {
        "checkpoint": str(checkpoint_path),
        "behavior": summarize_deliveries(deliveries, satisfaction_rates, checkpoint_path.name),
    }


def analyze_solo_completability(*, episodes: int, seed: int) -> dict[str, Any]:
    config = ResourceLogisticsConfig(
        deadline_min=12,
        deadline_max=20,
        depot_inventory_min=3,
        depot_inventory_max=5,
    )
    total = 0
    completable = 0
    travel_costs: list[int] = []
    deadlines: list[int] = []
    slacks: list[int] = []
    for episode in range(episodes):
        env = ResourceLogisticsEnv(config=config, seed=seed + episode)
        env.reset(seed=seed + episode)
        depot_costs = {depot: env.graph.shortest_costs(depot) for depot in env.graph.depots()}
        for demand in env.demands:
            stocked_depots = [
                depot
                for depot in env.graph.depots()
                if env.depot_inventory[depot][demand.resource_type] > 0
            ]
            nearest_cost = min(depot_costs[depot][demand.node_id] for depot in stocked_depots)
            total += 1
            travel_costs.append(nearest_cost)
            deadlines.append(demand.deadline)
            slacks.append(demand.deadline - nearest_cost)
            if nearest_cost <= demand.deadline:
                completable += 1
        env.close()
    return {
        "episodes": episodes,
        "demands": total,
        "solo_completable": completable,
        "solo_completable_fraction": safe_fraction(completable, total),
        "mean_nearest_depot_cost": float(np.mean(travel_costs)),
        "p50_nearest_depot_cost": percentile(travel_costs, 50),
        "p90_nearest_depot_cost": percentile(travel_costs, 90),
        "mean_deadline": float(np.mean(deadlines)),
        "mean_slack": float(np.mean(slacks)),
        "p10_slack": percentile(slacks, 10),
        "p50_slack": percentile(slacks, 50),
    }


def summarize_channel(slots: list[tuple[int, int, int]], label: str) -> dict[str, Any]:
    claim_values = [slot[0] for slot in slots]
    node_values = [slot[1] for slot in slots]
    payload_values = [slot[2] for slot in slots]
    claim_counts = Counter(claim_values)
    node_counts = Counter(node_values)
    payload_counts = Counter(payload_values)
    message_counts = Counter(slot[0] * 32 + slot[1] * 2 + slot[2] for slot in slots)
    claim_entropy = entropy_bits(claim_values)
    node_entropy = entropy_bits(node_values)
    payload_entropy = entropy_bits(payload_values)
    slot_entropy_total = claim_entropy + node_entropy + payload_entropy
    joint_entropy = entropy_from_counts(message_counts)
    non_null = sum(1 for slot in slots if slot[0] != NULL_CLAIM)
    non_default = sum(1 for slot in slots if slot != DEFAULT_MESSAGE)
    return {
        "label": label,
        "messages": len(slots),
        "non_null_fraction": safe_fraction(non_null, len(slots)),
        "non_default_fraction": safe_fraction(non_default, len(slots)),
        "claim_type_entropy_bits": claim_entropy,
        "node_id_entropy_bits": node_entropy,
        "payload_entropy_bits": payload_entropy,
        "slot_entropy_total_bits": slot_entropy_total,
        "joint_message_entropy_bits": joint_entropy,
        "claim_counts": {str(key): value for key, value in sorted(claim_counts.items())},
        "node_counts": {str(key): value for key, value in sorted(node_counts.items())},
        "payload_counts": {str(key): value for key, value in sorted(payload_counts.items())},
        "message_counts": {str(key): value for key, value in sorted(message_counts.items())},
        "top_claim_types": top_counts([CLAIM_TYPES[value] for value in claim_values], 8),
        "top_messages": top_counts([f"{slot[0]}:{slot[1]}:{slot[2]}" for slot in slots], 10),
    }


def summarize_deliveries(
    deliveries: list[DeliveryRecord],
    satisfaction_rates: list[float],
    label: str,
) -> dict[str, Any]:
    successful = len(deliveries)
    solo_successes = sum(delivery.solo_success for delivery in deliveries)
    coordinated = sum(delivery.coordinated for delivery in deliveries)
    transfers = sum(delivery.transferred for delivery in deliveries)
    transit_rendezvous = sum(delivery.transit_rendezvous for delivery in deliveries)
    unknown = sum(not delivery.provenance_known for delivery in deliveries)
    return {
        "label": label,
        "episodes": len(satisfaction_rates),
        "mean_satisfaction": float(np.mean(satisfaction_rates)) if satisfaction_rates else 0.0,
        "successful_deliveries": successful,
        "solo_success_fraction": safe_fraction(solo_successes, successful),
        "coordination_fraction": safe_fraction(coordinated, successful),
        "transfer_fraction": safe_fraction(transfers, successful),
        "transit_rendezvous_fraction": safe_fraction(transit_rendezvous, successful),
        "unknown_provenance_fraction": safe_fraction(unknown, successful),
    }


def aggregate_channel_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    claim_counts: Counter[int] = Counter()
    node_counts: Counter[int] = Counter()
    payload_counts: Counter[int] = Counter()
    message_counts: Counter[int] = Counter()
    for run in runs:
        channel = run["channel"]
        claim_counts.update(
            {int(key): int(value) for key, value in channel["claim_counts"].items()}
        )
        node_counts.update({int(key): int(value) for key, value in channel["node_counts"].items()})
        payload_counts.update(
            {int(key): int(value) for key, value in channel["payload_counts"].items()}
        )
        message_counts.update(
            {int(key): int(value) for key, value in channel["message_counts"].items()}
        )
    total_messages = sum(message_counts.values())
    non_null = sum(count for claim_type, count in claim_counts.items() if claim_type != NULL_CLAIM)
    non_default = sum(
        count
        for message_code, count in message_counts.items()
        if message_code != DEFAULT_MESSAGE[0] * 32 + DEFAULT_MESSAGE[1] * 2 + DEFAULT_MESSAGE[2]
    )
    claim_entropy = entropy_from_counts(claim_counts)
    node_entropy = entropy_from_counts(node_counts)
    payload_entropy = entropy_from_counts(payload_counts)
    return {
        "label": "aggregate",
        "messages": total_messages,
        "non_null_fraction": safe_fraction(non_null, total_messages),
        "non_default_fraction": safe_fraction(non_default, total_messages),
        "claim_type_entropy_bits": claim_entropy,
        "node_id_entropy_bits": node_entropy,
        "payload_entropy_bits": payload_entropy,
        "slot_entropy_total_bits": claim_entropy + node_entropy + payload_entropy,
        "joint_message_entropy_bits": entropy_from_counts(message_counts),
        "top_claim_types": top_counts_from_counter(
            Counter({CLAIM_TYPES[key]: value for key, value in claim_counts.items()}),
            8,
        ),
        "top_messages": top_counts_from_counter(
            Counter({decode_message_code(key): value for key, value in message_counts.items()}),
            10,
        ),
    }


def aggregate_policy_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_deliveries = sum(int(run["behavior"]["successful_deliveries"]) for run in runs)
    total_episodes = sum(int(run["behavior"]["episodes"]) for run in runs)
    aggregate = {
        "episodes": total_episodes,
        "successful_deliveries": total_deliveries,
        "mean_satisfaction": weighted_mean(
            [
                (run["behavior"]["mean_satisfaction"], int(run["behavior"]["episodes"]))
                for run in runs
            ]
        ),
    }
    for key in [
        "solo_success_fraction",
        "coordination_fraction",
        "transfer_fraction",
        "transit_rendezvous_fraction",
        "unknown_provenance_fraction",
    ]:
        aggregate[key] = weighted_mean(
            [(run["behavior"][key], int(run["behavior"]["successful_deliveries"])) for run in runs]
        )
    return aggregate


def choose_recommendation(
    *,
    empirical_bits: float,
    solo_completable_fraction: float,
    ippo_solo_success_fraction: float,
    ippo_coordination_fraction: float,
    symbolic_coordination_fraction: float,
) -> str:
    coordination_delta = symbolic_coordination_fraction - ippo_coordination_fraction
    coordination_similar = abs(coordination_delta) <= 0.05
    coordination_much_higher = coordination_delta >= 0.20
    if empirical_bits < 4.0 and coordination_similar:
        return (
            "channel undertrained — try longer training / higher entropy coef before "
            "recalibrating env"
        )
    if solo_completable_fraction > 0.70 or ippo_solo_success_fraction > 0.80:
        return "env undertuned — recalibrate per §3.7 priority order"
    if empirical_bits >= 7.0 and coordination_much_higher:
        return (
            "env is at a coordination ceiling — recalibrate by tightening deadlines or "
            "reducing depot inventory"
        )
    return "ambiguous — report numbers, ask before proceeding"


def render_report(data: dict[str, Any]) -> str:
    channel = data["channel_utilization"]["aggregate"]
    solo = data["solo_completability"]
    ippo = data["ippo_behavior"]["aggregate"]
    symbolic = data["symbolic_behavior"]["aggregate"]
    lines = [
        "# Phase 2 Gap Analysis",
        "",
        "Diagnostics for the failed §9 Step 2 symbolic MAPPO gate. No retraining was run; "
        "all policy diagnostics load the existing Phase 1 IPPO and Phase 2 symbolic checkpoints.",
        "",
        "## Inputs",
        "",
        (
            "- Symbolic checkpoints: 2 seeds x "
            f"{data['metadata']['symbolic_episodes_per_seed']} eval episodes"
        ),
        (
            "- IPPO checkpoints: 2 seeds x "
            f"{data['metadata']['ippo_episodes_per_seed']} eval episodes"
        ),
        (
            "- Env-only solo-completability sample: "
            f"{data['metadata']['solo_env_episodes']} episodes"
        ),
        "- Device: CPU",
        "",
        "## 1. Channel Utilization",
        "",
        (
            "| Scope | Messages | Non-null | Non-default | H(claim) | H(node) | "
            "H(payload) | Total slot entropy | Joint message entropy |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["channel_utilization"]["by_seed"]:
        lines.append(channel_row(row))
    lines.append(channel_row({"label": "aggregate", **channel}))
    lines.extend(
        [
            "",
            f"Capacity budget: {SYMBOLIC_CAPACITY_BITS} bits/message. "
            "The aggregate slot-sum empirical entropy is "
            f"{channel['slot_entropy_total_bits']:.2f} bits/message.",
            "",
            "Top aggregate claim/message counts are reported per seed in the JSON artifact "
            "`artifacts/phase2_gap_diagnostics.json`.",
            "",
            "## 2. Solo-Completability Of Demands",
            "",
            (
                "| Episodes | Demands | Solo-completable | Mean nearest-depot cost | "
                "P90 cost | Mean deadline | Mean slack | P10 slack |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {solo['episodes']} | {solo['demands']} | "
                f"{format_pct(solo['solo_completable_fraction'])} | "
                f"{solo['mean_nearest_depot_cost']:.2f} | "
                f"{solo['p90_nearest_depot_cost']:.1f} | "
                f"{solo['mean_deadline']:.2f} | {solo['mean_slack']:.2f} | "
                f"{solo['p10_slack']:.1f} |"
            ),
            "",
            (
                "Interpretation check: if this exceeds 70%, the env structurally permits "
                "too much solo play."
            ),
            "",
            "## 3. IPPO Behavior Analysis",
            "",
            (
                "| Scope | Episodes | Mean satisfaction | Successful deliveries | Solo-success | "
                "Coordination | Transfers | Transit rendezvous | Unknown provenance |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in data["ippo_behavior"]["by_seed"]:
        lines.append(behavior_row(row))
    lines.append(behavior_row({"label": "aggregate", **ippo}))
    lines.extend(
        [
            "",
            (
                "Solo-success means the delivered unit was picked up by the delivering "
                "agent at a depot, had no transfer, and had no same-depot co-location "
                "while the unit was carried."
            ),
            "",
            "## 4. Symbolic MAPPO Coordination Behavior",
            "",
            (
                "| Scope | Episodes | Mean satisfaction | Successful deliveries | Solo-success | "
                "Coordination | Transfers | Transit rendezvous | Unknown provenance |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in data["symbolic_behavior"]["by_seed"]:
        lines.append(behavior_row(row))
    lines.append(behavior_row({"label": "aggregate", **symbolic}))
    coordination_delta = symbolic["coordination_fraction"] - ippo["coordination_fraction"]
    lines.extend(
        [
            "",
            f"Symbolic coordination minus IPPO coordination: {coordination_delta * 100:.1f}pp.",
            "",
            "## Decision",
            "",
            f"- Empirical bits/message: {channel['slot_entropy_total_bits']:.2f}",
            f"- Solo-completable demand fraction: {format_pct(solo['solo_completable_fraction'])}",
            f"- IPPO solo-success fraction: {format_pct(ippo['solo_success_fraction'])}",
            f"- IPPO coordination fraction: {format_pct(ippo['coordination_fraction'])}",
            f"- Symbolic coordination fraction: {format_pct(symbolic['coordination_fraction'])}",
            "",
            f"Recommendation: {data['recommendation']}",
        ]
    )
    return "\n".join(lines)


def channel_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['label']} | {int(row['messages'])} | "
        f"{format_pct(row['non_null_fraction'])} | {format_pct(row['non_default_fraction'])} | "
        f"{row['claim_type_entropy_bits']:.2f} | {row['node_id_entropy_bits']:.2f} | "
        f"{row['payload_entropy_bits']:.2f} | {row['slot_entropy_total_bits']:.2f} | "
        f"{row['joint_message_entropy_bits']:.2f} |"
    )


def behavior_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['label']} | {int(row['episodes'])} | {format_pct(row['mean_satisfaction'])} | "
        f"{int(row['successful_deliveries'])} | {format_pct(row['solo_success_fraction'])} | "
        f"{format_pct(row['coordination_fraction'])} | {format_pct(row['transfer_fraction'])} | "
        f"{format_pct(row['transit_rendezvous_fraction'])} | "
        f"{format_pct(row['unknown_provenance_fraction'])} |"
    )


def stack_observations(
    observations: dict[str, np.ndarray],
    env: ResourceLogisticsEnv,
) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([observations[agent] for agent in env.possible_agents]),
        dtype=torch.float32,
        device=DEVICE,
    )


def stack_action_masks(
    infos: dict[str, dict[str, Any]],
    env: ResourceLogisticsEnv,
) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([infos[agent]["action_mask"] for agent in env.possible_agents]),
        dtype=torch.bool,
        device=DEVICE,
    )


def symbolic_actions(
    movement_actions: np.ndarray,
    message_embeddings: np.ndarray,
    message_slots: np.ndarray,
    env: ResourceLogisticsEnv,
) -> dict[str, dict[str, object]]:
    actions: dict[str, dict[str, object]] = {}
    for agent_index, agent in enumerate(env.possible_agents):
        receivers = [receiver for receiver in env.possible_agents if receiver != agent]
        actions[agent] = {
            "movement": int(movement_actions[agent_index]),
            "message_embeddings": {
                receiver: message_embeddings[agent_index].tolist() for receiver in receivers
            },
            "message_raw": {"slots": message_slots[agent_index].astype(int).tolist()},
        }
    return actions


def entropy_bits(values: list[int] | list[str]) -> float:
    if not values:
        return 0.0
    counts = np.array(list(Counter(values).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def entropy_from_counts(counts: Counter[int] | Counter[str]) -> float:
    if not counts:
        return 0.0
    values = np.array(list(counts.values()), dtype=np.float64)
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def top_counts(values: list[str], limit: int) -> list[dict[str, int | str]]:
    return [{"value": value, "count": count} for value, count in Counter(values).most_common(limit)]


def top_counts_from_counter(
    counter: Counter[str],
    limit: int,
) -> list[dict[str, int | str]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def decode_message_code(code: int) -> str:
    claim = code // 32
    node = (code % 32) // 2
    payload = code % 2
    return f"{claim}:{node}:{payload}"


def weighted_mean(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in values)
    if total_weight == 0:
        return 0.0
    return float(sum(value * weight for value, weight in values) / total_weight)


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), q)) if values else 0.0


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


if __name__ == "__main__":
    main()
