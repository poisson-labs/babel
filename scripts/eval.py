import json
from pathlib import Path

import yaml
from babel.algorithms.mappo import evaluate_symbolic_checkpoint
from babel.attacks.emergent_spoof import EmergentSpoofAttack
from babel.env.dynamics import ResourceLogisticsConfig


def main() -> None:
    print("Starting Phase 3 Evaluation Sweep...")

    with open("configs/sweep/phase3_eval.yaml") as f:
        cfg = yaml.safe_load(f)

    with open("configs/env/resource_logistics_v21.yaml") as f:
        env_dict = yaml.safe_load(f)

    # Base configuration
    env_cfg = ResourceLogisticsConfig(**env_dict)
    checkpoint_path = Path("artifacts/checkpoints/mappo_symbolic_seed_0.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    episodes = cfg["eval_episodes"]
    base_seed = cfg["eval_seed"]

    results = {}

    # Sweep over p_adv values
    for p_adv in cfg["p_adv_sweep"]:
        print(f"\nEvaluating with EmergentSpoofAttack at p_adv = {p_adv}")

        attack = EmergentSpoofAttack(p_adv=p_adv, seed=base_seed + 1000)

        eval_result = evaluate_symbolic_checkpoint(
            checkpoint_path=checkpoint_path,
            env_config=env_cfg,
            episodes=episodes,
            seed=base_seed,
            attack=attack,
            device="cpu",
        )

        mean_satisfaction = sum(eval_result.satisfaction_rates) / len(
            eval_result.satisfaction_rates
        )
        attack_stats = attack.report()

        print(f"-> Satisfaction Rate: {mean_satisfaction:.3f}")
        if p_adv > 0:
            print(f"-> Attacks Attempted: {attack_stats['attacks_attempted']}")
            print(f"-> Attacks Successful: {attack_stats['attacks_successful']}")

        results[f"p_adv_{p_adv}"] = {
            "p_adv": p_adv,
            "mean_satisfaction": mean_satisfaction,
            "episodes": episodes,
            "attacks_attempted": attack_stats["attacks_attempted"],
            "attacks_successful": attack_stats["attacks_successful"],
        }

    summary_path = Path("artifacts/phase3_eval_summary.json")
    with summary_path.open("w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
