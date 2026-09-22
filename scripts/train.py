from __future__ import annotations

import json
import os
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
from babel.algorithms.mappo import (
    MAPPOTrainConfig,
    SymbolicChannelConfig,
    evaluate_gate2,
    evaluate_symbolic_checkpoint,
    train_mappo_symbolic,
)
from babel.env.dynamics import ResourceLogisticsConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="algo/ippo")
def main(cfg: DictConfig) -> None:
    _load_dotenv_file()
    env_config = _env_config_from_cfg(cfg.env)
    if str(cfg.get("algorithm", "ippo")) == "mappo_symbolic":
        _run_mappo_symbolic(cfg, env_config)
        return
    _run_ippo(cfg, env_config)


def _run_ippo(cfg: DictConfig, env_config: ResourceLogisticsConfig) -> None:
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


def _run_mappo_symbolic(cfg: DictConfig, env_config: ResourceLogisticsConfig) -> None:
    channel_type = cfg.channel.get("channel_type", "symbolic")
    channel_config = dict(cfg.channel)
    checkpoint_paths: list[Path] = []
    train_results: list[dict[str, Any]] = []

    if cfg.mode in {"train", "train_eval"}:
        for seed in list(cfg.seeds):
            train_config = _mappo_train_config_from_cfg(cfg, int(seed))
            result = train_mappo_symbolic(
                env_config=env_config,
                train_config=train_config,
                channel_type=channel_type,
                channel_config=channel_config,
            )
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
            result = evaluate_symbolic_checkpoint(
                checkpoint_path=checkpoint_path,
                env_config=env_config,
                episodes=int(cfg.eval_episodes),
                seed=int(cfg.eval_seed) + index * int(cfg.eval_episodes),
                device=str(cfg.device),
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
        gate = evaluate_gate2(
            all_satisfaction_rates,
            seed=int(cfg.eval_seed),
            resamples=int(cfg.bootstrap_resamples),
            ippo_point_estimate=float(cfg.ippo_point_estimate),
            ippo_upper_ci=float(cfg.ippo_upper_ci),
            gate_low=float(cfg.gate_low),
            gate_high=float(cfg.gate_high),
            borderline_pass_low=float(cfg.borderline_pass_low),
        )
        summary["gate"] = {
            **asdict(gate.ci),
            "episodes": len(all_satisfaction_rates),
            "ippo_point_estimate": float(cfg.ippo_point_estimate),
            "ippo_upper_ci": float(cfg.ippo_upper_ci),
            "gap_vs_ippo_point": gate.ippo_point_gap,
            "conservative_gap_vs_ippo_upper_ci": gate.ippo_upper_ci_gap,
            "status": gate.status,
            "passed": gate.passed,
            "gate_low": float(cfg.gate_low),
            "gate_high": float(cfg.gate_high),
            "borderline_pass_low": float(cfg.borderline_pass_low),
        }
        print(
            "Phase 2 symbolic MAPPO demand satisfaction: "
            f"{gate.ci.mean:.3f} (95% CI {gate.ci.low:.3f}-{gate.ci.high:.3f}, "
            f"n={len(all_satisfaction_rates)})"
        )
        print(
            "Gap vs IPPO point estimate: "
            f"{gate.ippo_point_gap * 100:.1f}pp; conservative gap vs IPPO upper CI: "
            f"{gate.ippo_upper_ci_gap * 100:.1f}pp"
        )
        if gate.status == "passed":
            print("✅ GATE 2 PASSED")
        elif gate.status == "borderline_pass_gap_ok_below_band":
            print(
                "BORDERLINE-PASS: gap is >=30pp, but mean is below the original "
                "[70%, 85%] band. Ask for confirmation before declaring Gate 2 passed."
            )
        elif gate.status == "borderline_gap_20_to_30pp":
            print(
                "BORDERLINE: communication gap is 20-30pp. Do not declare Gate 2 passed; "
                "ask whether to proceed or recalibrate."
            )
        elif gate.status == "fail_comm_gap_lt_20pp":
            print(
                "GATE 2 FAILED: gap is <20pp. The env does not reward communication enough; "
                "return to Step 1 calibration."
            )
        else:
            print("GATE 2 FAILED: mean is outside the target band.")

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
        num_resource_types=int(data.get("num_resource_types", 2)),
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
        message_dim=int(data.get("message_dim", 0)),
        message_history_length=int(data.get("message_history_length", 5)),
        private_demands=bool(data.get("private_demands", False)),
        discovery_reward=float(data.get("discovery_reward", 0.1)),
        role_asymmetry=bool(data.get("role_asymmetry", False)),
        semantic_channel=bool(data.get("semantic_channel", False)),
        replay_path=data["replay_path"],
        # V22 additions
        num_depots=int(data.get("num_depots", 3)),
        num_demand_nodes=int(data.get("num_demand_nodes", 6)),
        num_transit_hubs=int(data.get("num_transit_hubs", 3)),
        fuel_budget=int(data["fuel_budget"]) if data.get("fuel_budget") is not None else None,
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


def _symbolic_channel_config_from_cfg(cfg: DictConfig) -> SymbolicChannelConfig:
    slot_sizes_raw = cfg.get("channel_slot_sizes", None)
    slot_sizes = tuple(int(s) for s in slot_sizes_raw) if slot_sizes_raw else (8, 16, 2)
    return SymbolicChannelConfig(
        message_dim=int(cfg.channel.message_dim),
        hidden_dim=int(cfg.channel.hidden_dim),
        initial_temperature=float(cfg.channel.initial_temperature),
        min_temperature=float(cfg.channel.min_temperature),
        anneal_steps=int(cfg.channel.anneal_steps),
        slot_sizes=slot_sizes,
    )


def _mappo_train_config_from_cfg(cfg: DictConfig, seed: int) -> MAPPOTrainConfig:
    return MAPPOTrainConfig(
        total_timesteps=int(cfg.total_timesteps),
        num_envs=int(cfg.num_envs),
        rollout_length=int(cfg.rollout_length),
        ppo_epochs=int(cfg.ppo_epochs),
        minibatch_size=int(cfg.minibatch_size),
        learning_rate=float(cfg.learning_rate),
        channel_learning_rate=float(cfg.channel_learning_rate),
        clip_coef=float(cfg.clip_coef),
        gae_lambda=float(cfg.gae_lambda),
        gamma=float(cfg.gamma),
        entropy_coef=float(cfg.entropy_coef),
        value_coef=float(cfg.value_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        seed=seed,
        device=str(cfg.device),
        checkpoint_dir=Path(to_absolute_path(str(cfg.checkpoint_dir))),
        checkpoint_name=str(cfg.checkpoint_name)
        if cfg.get("checkpoint_name")
        else f"mappo_symbolic_seed_{seed}.pt",
        wandb_project=str(cfg.wandb.project),
        wandb_group=str(cfg.wandb.group),
        wandb_mode=str(cfg.wandb.mode),
        track_wandb=bool(cfg.wandb.enabled),
        log_interval_updates=int(cfg.log_interval_updates),
        load_checkpoint=str(cfg.load_checkpoint) if cfg.get("load_checkpoint") else None,
    )


def _load_dotenv_file() -> None:
    env_path = Path(to_absolute_path(".env"))
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    main()
