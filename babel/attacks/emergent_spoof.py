from typing import Any

import numpy as np
import torch

from babel.attacks.base import CommsAttack
from babel.channels.symbolic import SymbolicBatch, SymbolicChannel
from babel.env.dynamics import MAX_VISIBLE_DEMANDS, ResourceLogisticsEnv


class EmergentSpoofAttack(CommsAttack):
    """
    Attack 4c: Capability-matched semantic adversary (Emergent Spoof).
    Because MARL policies develop arbitrary emergent languages,
    we cannot manually flip discrete slots.
    Instead, we hallucinate a fake demand in the Scout's observation and feed it through the
    Scout's frozen actor to perfectly translate the semantic lie into the emergent 10-bit space.

    Active Misdirection Attack:
    Rather than zeroing out demands (which is a "silence" signal),
    we actively mislead the Deliverers
    by injecting a fake demand at a demand node that has no active demand.
    """

    def __init__(self, p_adv: float, seed: int = 42):
        self.p_adv = p_adv
        self.rng = np.random.default_rng(seed)
        self.attempts = 0
        self.successes = 0

    def perturb_intents(
        self,
        actor: Any,
        obs_tensor: torch.Tensor,
        hidden_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        env: ResourceLogisticsEnv,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.p_adv <= 0.0:
            return None

        cfg = env.config
        batch_size = obs_tensor.shape[0]
        fake_obs = obs_tensor.clone()
        did_perturb = False

        # Compute observation layout dynamically
        # V22 uses fixed observation space where node_id is in demand features
        # (size = num_nodes + num_resource_types + 2)
        # Check if the environment config uses node_id in demand features
        if hasattr(cfg, "num_nodes") and cfg.num_nodes > 12:
            demand_feature_width = cfg.num_nodes + cfg.num_resource_types + 2
            is_v22 = True
        else:
            demand_feature_width = cfg.num_resource_types + 2
            is_v22 = False

        demand_start = (
            cfg.num_nodes  # position one-hot
            + cfg.num_resource_types  # inventory
            + 8 * 4  # MAX_NEIGHBORHOOD_NODES * 4 (neighborhood features)
            + cfg.num_resource_types  # depot inventory
        )
        demand_length = MAX_VISIBLE_DEMANDS * demand_feature_width

        # Commitment features come after: demands + message_history + step_counter
        commitment_start = cfg.message_history_offset
        if cfg.message_dim > 0:
            commitment_start += cfg.message_history_length * (cfg.num_agents - 1) * cfg.message_dim
        commitment_start += 1  # step counter

        for i in range(batch_size):
            agent_name = env.possible_agents[i]
            is_scout = cfg.role_asymmetry and agent_name in ["agent_0", "agent_1"]

            if is_scout and self.rng.random() < self.p_adv:
                self.attempts += 1

                # Zero out all demand features
                fake_obs[i, demand_start : demand_start + demand_length] = 0.0

                # Zero out commitment features so Scout forgets what it was doing
                fake_obs[i, commitment_start : commitment_start + cfg.max_demands] = 0.0

                if is_v22:
                    # Inject a fake demand at a demand node that has no active demand
                    active_nodes = {d.node_id for d in env.demands if d.status == "active"}
                    candidate_nodes = [
                        node_id
                        for node_id, node_type in enumerate(env.graph.node_types)
                        if node_type == "demand" and node_id not in active_nodes
                    ]
                    if not candidate_nodes:
                        candidate_nodes = [
                            node_id
                            for node_id in range(cfg.num_nodes)
                            if node_id not in active_nodes
                        ]
                    if not candidate_nodes:
                        candidate_nodes = [0]

                    fake_node_id = self.rng.choice(candidate_nodes)
                    fake_resource_type = self.rng.choice(cfg.num_resource_types)

                    # Create fake demand vector
                    fake_demand = np.zeros(demand_feature_width, dtype=np.float32)
                    fake_demand[fake_node_id] = 1.0
                    fake_demand[cfg.num_nodes + fake_resource_type] = 1.0
                    fake_demand[cfg.num_nodes + cfg.num_resource_types] = (
                        1.0  # remaining time: max (1.0)
                    )
                    fake_demand[cfg.num_nodes + cfg.num_resource_types + 1] = (
                        1.0  # reward: max (1.0)
                    )

                    fake_demand_tensor = torch.from_numpy(fake_demand).to(
                        device=fake_obs.device, dtype=fake_obs.dtype
                    )
                    fake_obs[i, demand_start : demand_start + demand_feature_width] = (
                        fake_demand_tensor
                    )

                self.successes += 1
                did_perturb = True

        if not did_perturb:
            return None

        # Pass the fake observation through the frozen actor to get the fake intent
        with torch.no_grad():
            _, _, fake_intents = actor.forward_step(fake_obs, hidden_tensor, mask_tensor)

        return fake_intents, fake_obs

    def perturb(
        self,
        batch: SymbolicBatch,
        channel: SymbolicChannel,
        env: ResourceLogisticsEnv,
        step: int,
    ) -> SymbolicBatch:
        # Legacy method for base.py compatibility, but we use perturb_intents now
        return batch

    def report(self) -> dict[str, float]:
        return {
            "attacks_attempted": float(self.attempts),
            "attacks_successful": float(self.successes),
        }
