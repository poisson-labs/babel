"""
Monte Adversarial Sweep — Phase 3
=================================
Evaluates the MAPPO V22c checkpoint under the EmergentSpoofAttack
across a range of adversarial probabilities (p_adv).

The thesis: if the communication channel is load-bearing, Monte should be
able to collapse MAPPO's 22.2% satisfaction back to the IPPO baseline of
11.8% by corrupting the semantic content of the Scout's messages.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from babel.algorithms.mappo import evaluate_symbolic_checkpoint
from babel.attacks.emergent_spoof import EmergentSpoofAttack
from babel.env.dynamics import ResourceLogisticsConfig


@dataclass
class MonteResult:
    p_adv: float
    mean_satisfaction: float
    ci_low: float
    ci_high: float
    mean_reward: float
    attacks_attempted: int
    attacks_successful: int
    episodes: int
    elapsed_seconds: float


def run_monte_sweep(
    *,
    checkpoint_path: Path,
    env_config: ResourceLogisticsConfig,
    p_adv_values: list[float],
    episodes_per_point: int = 200,
    seed: int = 500000,
) -> list[MonteResult]:
    results: list[MonteResult] = []

    for p_adv in p_adv_values:
        attack = EmergentSpoofAttack(p_adv=p_adv, seed=seed) if p_adv > 0 else None

        print(f"\n{'=' * 60}")
        print(f"  Monte sweep: p_adv = {p_adv:.2f}")
        print(f"{'=' * 60}")

        t0 = time.perf_counter()
        result = evaluate_symbolic_checkpoint(
            checkpoint_path=checkpoint_path,
            env_config=env_config,
            episodes=episodes_per_point,
            seed=seed,
            attack=attack,
            device="cpu",
        )
        elapsed = time.perf_counter() - t0

        # Bootstrap CI
        rng = np.random.default_rng(seed)
        values = np.array(result.satisfaction_rates)
        bootstrap_means = np.array(
            [
                float(np.mean(rng.choice(values, size=len(values), replace=True)))
                for _ in range(1000)
            ]
        )
        ci_low = float(np.quantile(bootstrap_means, 0.025))
        ci_high = float(np.quantile(bootstrap_means, 0.975))

        report = attack.report() if attack else {"attacks_attempted": 0, "attacks_successful": 0}

        monte_result = MonteResult(
            p_adv=p_adv,
            mean_satisfaction=result.mean_satisfaction,
            ci_low=ci_low,
            ci_high=ci_high,
            mean_reward=result.mean_reward,
            attacks_attempted=int(report["attacks_attempted"]),
            attacks_successful=int(report["attacks_successful"]),
            episodes=episodes_per_point,
            elapsed_seconds=elapsed,
        )
        results.append(monte_result)

        print(
            f"  Satisfaction: {monte_result.mean_satisfaction:.4f} (CI: {ci_low:.4f}-{ci_high:.4f})"
        )
        print(f"  Reward: {monte_result.mean_reward:.2f}")
        print(
            f"  Attacks: {monte_result.attacks_attempted} attempted, "
            f"{monte_result.attacks_successful} successful"
        )
        print(f"  Elapsed: {elapsed:.1f}s")

    return results


def main() -> None:
    with open("configs/env/resource_logistics_v22c.yaml") as f:
        env_config = ResourceLogisticsConfig(**yaml.safe_load(f))

    checkpoint_path = Path("artifacts/checkpoints/mappo_symbolic_seed_0.pt")

    # Full sweep: 0% to 100% adversarial probability in 10% increments
    p_adv_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    ippo_baseline = 0.1180  # V22c IPPO result
    ippo_upper_ci = 0.1330

    print("=" * 60)
    print("  MONTE ADVERSARIAL SWEEP — Phase 3")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  IPPO baseline: {ippo_baseline:.4f} (upper CI: {ippo_upper_ci:.4f})")
    print(f"  Sweep points: {len(p_adv_values)}")
    print("  Episodes per point: 200")
    print("=" * 60)

    results = run_monte_sweep(
        checkpoint_path=checkpoint_path,
        env_config=env_config,
        p_adv_values=p_adv_values,
        episodes_per_point=200,
        seed=500000,
    )

    # Summary table
    print("\n" + "=" * 80)
    print("  MONTE SWEEP RESULTS")
    print("=" * 80)
    print(f"  {'p_adv':>6}  {'Satisfaction':>12}  {'95% CI':>16}  {'vs IPPO':>10}  {'Attacks':>10}")
    print("-" * 80)

    for r in results:
        gap = r.mean_satisfaction - ippo_baseline
        gap_str = f"+{gap * 100:.1f}pp" if gap > 0 else f"{gap * 100:.1f}pp"
        ci_str = f"{r.ci_low:.4f}-{r.ci_high:.4f}"
        print(
            f"  {r.p_adv:>6.2f}  {r.mean_satisfaction:>12.4f}  "
            f"{ci_str:>16}  {gap_str:>10}  {r.attacks_attempted:>10}"
        )

    # Did Monte collapse the gap?
    clean = next(r for r in results if r.p_adv == 0.0)
    full = next(r for r in results if r.p_adv == 1.0)
    comm_premium = clean.mean_satisfaction - ippo_baseline
    residual = full.mean_satisfaction - ippo_baseline
    premium_destroyed = 1.0 - (residual / comm_premium) if comm_premium > 0 else 0.0

    print(f"\n  Communication premium (p_adv=0): +{comm_premium * 100:.1f}pp")
    print(
        f"  Residual after full attack (p_adv=1): "
        f"{'+' if residual > 0 else ''}{residual * 100:.1f}pp"
    )
    print(f"  Premium destroyed by Monte: {premium_destroyed * 100:.1f}%")

    if full.mean_satisfaction <= ippo_upper_ci:
        print("\n  ✅ MONTE COLLAPSED the swarm to IPPO baseline.")
        print("     The communication channel is LOAD-BEARING.")
    elif full.mean_satisfaction <= ippo_baseline + 0.05:
        print("\n  ✅ MONTE SEVERELY DEGRADED the swarm (within 5pp of IPPO).")
        print("     The communication channel is LOAD-BEARING.")
    else:
        print("\n  ⚠ Monte reduced performance but swarm retained partial resilience.")
        print(f"    Full-attack satisfaction: {full.mean_satisfaction:.4f}")

    # Save results
    output = {
        "ippo_baseline": ippo_baseline,
        "ippo_upper_ci": ippo_upper_ci,
        "checkpoint": str(checkpoint_path),
        "comm_premium_pp": comm_premium * 100,
        "residual_pp": residual * 100,
        "premium_destroyed_pct": premium_destroyed * 100,
        "sweep": [
            {
                "p_adv": r.p_adv,
                "satisfaction": r.mean_satisfaction,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "reward": r.mean_reward,
                "attacks_attempted": r.attacks_attempted,
                "attacks_successful": r.attacks_successful,
            }
            for r in results
        ],
    }
    out_path = Path("artifacts/monte_v22c_sweep.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n  Wrote results to {out_path}")


if __name__ == "__main__":
    main()
