from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
from babel.algorithms.ippo import (
    IPPOTrainConfig,
    bootstrap_success_ci,
    evaluate_checkpoint,
    train_ippo,
)
from babel.env.dynamics import ResourceLogisticsConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="algo/ippo")
def main(cfg: DictConfig) -> None:
    env_config = _env_config_from_cfg(cfg.env)
    checkpoint_paths: list[Path] = []
    train_results: list[dict[str, Any]] = []

    if cfg.mode in {"train", "train_eval"}:
        for seed in list(cfg.seeds):
            train_config = _train_config_from_cfg(cfg, int(seed))
            result = train_ippo(env_config=env_config, train_config=train_config)
            checkpoint_paths.append(result.checkpoint_path)
            train_results.append(
                {
                    "seed": result.seed,
                    "checkpoint_path": str(result.checkpoint_path),
                    "global_step": result.global_step,
                    "updates": result.updates,
                    "elapsed_seconds": result.elapsed_seconds,
                    "recent_satisfaction": result.recent_satisfaction,
                }
            )

    if cfg.mode == "eval":
        checkpoint_paths = [
            Path(to_absolute_path(path)) for path in list(cfg.eval_checkpoint_paths)
        ]

    eval_results: list[dict[str, Any]] = []
    all_satisfaction_rates: list[float] = []
    if cfg.mode in {"eval", "train_eval"}:
        for index, checkpoint_path in enumerate(checkpoint_paths):
            replay_dir = (
                Path(to_absolute_path(str(cfg.eval_replay_dir))) / f"seed_{index}"
                if bool(cfg.save_eval_replays)
                else None
            )
            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                env_config=env_config,
                episodes=int(cfg.eval_episodes),
                seed=int(cfg.eval_seed) + index * int(cfg.eval_episodes),
                device=str(cfg.device),
                replay_dir=replay_dir,
            )
            all_satisfaction_rates.extend(result.satisfaction_rates)
            eval_results.append(
                {
                    "checkpoint_path": str(result.checkpoint_path),
                    "episodes": result.episodes,
                    "mean_satisfaction": result.mean_satisfaction,
                    "mean_reward": result.mean_reward,
                }
            )

    summary: dict[str, Any] = {
        "mode": str(cfg.mode),
        "train": train_results,
        "eval": eval_results,
    }
    if all_satisfaction_rates:
        ci = bootstrap_success_ci(
            all_satisfaction_rates,
            seed=int(cfg.eval_seed),
            resamples=int(cfg.bootstrap_resamples),
        )
        summary["gate"] = {
            **asdict(ci),
            "episodes": len(all_satisfaction_rates),
            "passed": bool(float(cfg.gate_low) <= ci.mean <= float(cfg.gate_high)),
            "gate_low": float(cfg.gate_low),
            "gate_high": float(cfg.gate_high),
        }
        print(
            "Phase 1 IPPO demand satisfaction: "
            f"{ci.mean:.3f} (95% CI {ci.low:.3f}-{ci.high:.3f}, "
            f"n={len(all_satisfaction_rates)})"
        )
        if summary["gate"]["passed"]:
            print("✅ GATE 1 PASSED")
        elif ci.mean < float(cfg.gate_low):
            print("GATE 1 FAILED: environment is too hard under §3.7.")
        else:
            print("GATE 1 FAILED: environment is too easy under §3.7.")

    summary_path = Path(to_absolute_path(str(cfg.summary_path)))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary to {summary_path}")


def _env_config_from_cfg(cfg: DictConfig) -> ResourceLogisticsConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("env config must resolve to a mapping")
    return ResourceLogisticsConfig(
        num_nodes=int(data["num_nodes"]),
        num_agents=int(data["num_agents"]),
        episode_length=int(data["episode_length"]),
        inventory_capacity=int(data["inventory_capacity"]),
        initial_demands_min=int(data["initial_demands_min"]),
        initial_demands_max=int(data["initial_demands_max"]),
        deadline_min=int(data["deadline_min"]),
        deadline_max=int(data["deadline_max"]),
        reward_values=tuple(int(value) for value in data["reward_values"]),
        depot_inventory_min=int(data["depot_inventory_min"]),
        depot_inventory_max=int(data["depot_inventory_max"]),
        replenishment_rate=float(data["replenishment_rate"]),
        max_demands=int(data["max_demands"]),
        replay_path=data["replay_path"],
    )


def _train_config_from_cfg(cfg: DictConfig, seed: int) -> IPPOTrainConfig:
    return IPPOTrainConfig(
        total_timesteps=int(cfg.total_timesteps),
        num_envs=int(cfg.num_envs),
        rollout_length=int(cfg.rollout_length),
        ppo_epochs=int(cfg.ppo_epochs),
        minibatch_size=int(cfg.minibatch_size),
        learning_rate=float(cfg.learning_rate),
        clip_coef=float(cfg.clip_coef),
        gae_lambda=float(cfg.gae_lambda),
        gamma=float(cfg.gamma),
        entropy_coef=float(cfg.entropy_coef),
        value_coef=float(cfg.value_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        seed=seed,
        device=str(cfg.device),
        checkpoint_dir=Path(to_absolute_path(str(cfg.checkpoint_dir))),
        checkpoint_name=f"ippo_seed_{seed}.pt",
        wandb_project=str(cfg.wandb.project),
        wandb_group=str(cfg.wandb.group),
        wandb_mode=str(cfg.wandb.mode),
        track_wandb=bool(cfg.wandb.enabled),
        log_interval_updates=int(cfg.log_interval_updates),
    )


if __name__ == "__main__":
    main()
