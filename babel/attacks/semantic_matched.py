import numpy as np

from babel.attacks.base import CommsAttack
from babel.channels.symbolic import SymbolicBatch, SymbolicChannel
from babel.env.dynamics import ResourceLogisticsEnv


class SymbolicSpoofAttack(CommsAttack):
    def __init__(self, p_adv: float, seed: int = 42):
        self.p_adv = p_adv
        self.rng = np.random.default_rng(seed)
        self.attempts = 0
        self.successes = 0

    def perturb(
        self,
        batch: SymbolicBatch,
        channel: SymbolicChannel,
        env: ResourceLogisticsEnv,
        step: int,
    ) -> SymbolicBatch:
        if self.p_adv <= 0.0:
            return batch

        slots = batch.slots.clone()
        num_agents = slots.shape[0]

        for i in range(num_agents):
            if self.rng.random() < self.p_adv:
                self.attempts += 1

                # claim_type is slot 0, node_id is slot 1, payload is slot 2
                _claim_type = int(slots[i, 0].item())
                _node_id = int(slots[i, 1].item())
                resource_type = int(slots[i, 2].item())

                # The network ignores node_id because the Scout cannot observe it.
                # The network uses the channel purely to coordinate WHICH resource to pick up!
                # To break the swarm, we must spoof the resource payload (slot 2).

                # Flip the resource type to a different random type
                n_types = max(env.config.num_resource_types, 2)
                alternatives = [t for t in range(n_types) if t != resource_type]
                spoofed_resource = int(self.rng.choice(alternatives))
                slots[i, 2] = spoofed_resource
                self.successes += 1

        # Re-decode the perturbed slots into embeddings
        embeddings = channel.decode_slots(slots)

        return SymbolicBatch(
            slots=slots, embeddings=embeddings, logprobs=batch.logprobs, entropy=batch.entropy
        )

    def report(self) -> dict[str, float]:
        return {
            "attacks_attempted": float(self.attempts),
            "attacks_successful": float(self.successes),
        }
