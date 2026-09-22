# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from babel.env.dynamics import (
    ResourceLogisticsConfig,
    ResourceLogisticsEnv,
)
from babel.networks.actor import IPPOActor
from babel.networks.critic import IPPOCritic


@dataclass(frozen=True, slots=True)
class IPPOTrainConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 16
    rollout_length: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: Path | str = "artifacts/checkpoints"
    checkpoint_name: str | None = None
    wandb_project: str = "babel"
    wandb_group: str = "phase1_ippo"
    wandb_mode: str = "online"
    track_wandb: bool = True
    log_interval_updates: int = 10


@dataclass(frozen=True, slots=True)
class TrainingResult:
    seed: int
    checkpoint_path: Path
    global_step: int
    updates: int
    elapsed_seconds: float
    recent_satisfaction: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    checkpoint_path: Path
    episodes: int
    satisfaction_rates: list[float]
    rewards: list[float]

    @property
    def mean_satisfaction(self) -> float:
        return float(np.mean(self.satisfaction_rates)) if self.satisfaction_rates else 0.0

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    mean: float
    low: float
    high: float


def train_ippo(
    *,
    env_config: ResourceLogisticsConfig,
    train_config: IPPOTrainConfig,
) -> TrainingResult:
    set_global_seeds(train_config.seed)
    device = torch.device(train_config.device)
    envs = [
        ResourceLogisticsEnv(
            config=replace(env_config, replay_path=None), seed=train_config.seed + env_id
        )
        for env_id in range(train_config.num_envs)
    ]
    current_seed = train_config.seed + train_config.num_envs
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, dict[str, Any]]] = []
    for env_id, env in enumerate(envs):
        obs, info = env.reset(seed=train_config.seed + env_id)
        observations.append(obs)
        infos.append(info)

    observation_dim = env_config.observation_dim
    num_agents = env_config.num_agents
    action_dim_val = env_config.action_dim
    actor = IPPOActor(observation_dim=observation_dim, action_dim=action_dim_val).to(device)
    critic = IPPOCritic(observation_dim=observation_dim).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=train_config.learning_rate,
        eps=1e-5,
    )
    batch_size = train_config.num_envs * num_agents
    rollout_batch_size = train_config.rollout_length * batch_size
    minibatch_size = min(train_config.minibatch_size, rollout_batch_size)
    updates = max(ceil(train_config.total_timesteps / rollout_batch_size), 1)

    actor_hidden = actor.initial_hidden(batch_size, device)
    critic_hidden = critic.initial_hidden(batch_size, device)

    run = _start_wandb(train_config, env_config)
    global_step = 0
    episode_satisfaction: list[float] = []
    start_time = time.perf_counter()

    for update in range(1, updates + 1):
        obs_buf = torch.zeros(
            (train_config.rollout_length, batch_size, observation_dim), device=device
        )
        mask_buf = torch.zeros(
            (train_config.rollout_length, batch_size, action_dim_val),
            dtype=torch.bool,
            device=device,
        )
        action_buf = torch.zeros(
            (train_config.rollout_length, batch_size), dtype=torch.long, device=device
        )
        logprob_buf = torch.zeros((train_config.rollout_length, batch_size), device=device)
        reward_buf = torch.zeros((train_config.rollout_length, batch_size), device=device)
        done_buf = torch.zeros((train_config.rollout_length, batch_size), device=device)
        value_buf = torch.zeros((train_config.rollout_length, batch_size), device=device)
        actor_hidden_buf = torch.zeros(
            (train_config.rollout_length, batch_size, actor.hidden_dim),
            device=device,
        )
        critic_hidden_buf = torch.zeros(
            (train_config.rollout_length, batch_size, critic.hidden_dim),
            device=device,
        )

        for step in range(train_config.rollout_length):
            obs_tensor = _stack_observations(observations, device)
            mask_tensor = _stack_action_masks(infos, device)

            obs_buf[step] = obs_tensor
            mask_buf[step] = mask_tensor
            actor_hidden_buf[step] = actor_hidden
            critic_hidden_buf[step] = critic_hidden

            with torch.no_grad():
                logits, next_actor_hidden = actor.forward_step(
                    obs_tensor, actor_hidden, mask_tensor
                )
                values, next_critic_hidden = critic.forward_step(obs_tensor, critic_hidden)
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                logprobs = distribution.log_prob(actions)

            action_buf[step] = actions
            logprob_buf[step] = logprobs
            value_buf[step] = values

            env_actions = _unflatten_actions(actions.cpu().numpy(), envs)
            next_observations: list[dict[str, np.ndarray]] = []
            next_infos: list[dict[str, dict[str, Any]]] = []
            done_flags = np.zeros((train_config.num_envs, num_agents), dtype=np.float32)

            for env_id, env in enumerate(envs):
                obs, rewards, terminations, truncations, info = env.step(env_actions[env_id])
                done = all(terminations.values()) or all(truncations.values())
                reward_values = [rewards[agent] for agent in env.possible_agents]
                reward_buf[step, env_id * num_agents : (env_id + 1) * num_agents] = torch.tensor(
                    reward_values,
                    dtype=torch.float32,
                    device=device,
                )
                if done:
                    done_flags[env_id, :] = 1.0
                    outcome = cast(
                        dict[str, Any], next(iter(info.values())).get("outcomes", env.outcomes())
                    )
                    episode_satisfaction.append(float(outcome["demand_satisfaction_rate"]))
                    obs, info = env.reset(seed=current_seed)
                    current_seed += 1
                    next_actor_hidden[env_id * num_agents : (env_id + 1) * num_agents] = 0.0
                    next_critic_hidden[env_id * num_agents : (env_id + 1) * num_agents] = 0.0
                next_observations.append(obs)
                next_infos.append(info)

            done_buf[step] = torch.tensor(
                done_flags.reshape(-1), dtype=torch.float32, device=device
            )
            observations = next_observations
            infos = next_infos
            actor_hidden = next_actor_hidden.detach()
            critic_hidden = next_critic_hidden.detach()
            global_step += batch_size

        with torch.no_grad():
            next_obs_tensor = _stack_observations(observations, device)
            next_values, _ = critic.forward_step(next_obs_tensor, critic_hidden)
            advantages = _compute_gae(
                rewards=reward_buf,
                dones=done_buf,
                values=value_buf,
                next_value=next_values,
                gamma=train_config.gamma,
                gae_lambda=train_config.gae_lambda,
            )
            returns = advantages + value_buf

        flat_obs = obs_buf.reshape((-1, observation_dim))
        flat_masks = mask_buf.reshape((-1, action_dim_val))
        flat_actions = action_buf.reshape(-1)
        flat_logprobs = logprob_buf.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = value_buf.reshape(-1)
        flat_actor_hidden = actor_hidden_buf.reshape((-1, actor.hidden_dim))
        flat_critic_hidden = critic_hidden_buf.reshape((-1, critic.hidden_dim))
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        indices = np.arange(rollout_batch_size)
        last_metrics: dict[str, float] = {}
        for _epoch in range(train_config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_batch_size, minibatch_size):
                mb_indices = torch.as_tensor(
                    indices[start : start + minibatch_size], dtype=torch.long, device=device
                )
                new_logits, _ = actor.forward_step(
                    flat_obs[mb_indices],
                    flat_actor_hidden[mb_indices],
                    flat_masks[mb_indices],
                )
                new_values, _ = critic.forward_step(
                    flat_obs[mb_indices], flat_critic_hidden[mb_indices]
                )
                distribution = Categorical(logits=new_logits)
                new_logprobs = distribution.log_prob(flat_actions[mb_indices])
                entropy = distribution.entropy().mean()
                logratio = new_logprobs - flat_logprobs[mb_indices]
                ratio = logratio.exp()

                pg_loss_1 = -flat_advantages[mb_indices] * ratio
                pg_loss_2 = -flat_advantages[mb_indices] * torch.clamp(
                    ratio,
                    1.0 - train_config.clip_coef,
                    1.0 + train_config.clip_coef,
                )
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                value_loss = 0.5 * ((new_values - flat_returns[mb_indices]) ** 2).mean()
                entropy_loss = -train_config.entropy_coef * entropy
                loss = pg_loss + train_config.value_coef * value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    train_config.max_grad_norm,
                )
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clipped = (
                        (torch.abs(ratio - 1.0) > train_config.clip_coef).float().mean().item()
                    )
                last_metrics = {
                    "loss/total": float(loss.item()),
                    "loss/policy": float(pg_loss.item()),
                    "loss/value": float(value_loss.item()),
                    "loss/entropy": float(entropy.item()),
                    "loss/approx_kl": float(approx_kl),
                    "loss/clip_fraction": float(clipped),
                    "charts/learning_rate": train_config.learning_rate,
                    "charts/global_step": float(global_step),
                    "charts/recent_satisfaction": _recent_mean(episode_satisfaction),
                    "charts/value_mean": float(flat_values.mean().item()),
                }

        if run is not None and (update % train_config.log_interval_updates == 0 or update == 1):
            run.log(last_metrics, step=global_step)

    checkpoint_path = _save_checkpoint(
        actor=actor,
        critic=critic,
        env_config=env_config,
        train_config=train_config,
        global_step=global_step,
    )
    if run is not None:
        run.log({"charts/final_global_step": global_step}, step=global_step)
        run.finish()
    for env in envs:
        env.close()

    return TrainingResult(
        seed=train_config.seed,
        checkpoint_path=checkpoint_path,
        global_step=global_step,
        updates=updates,
        elapsed_seconds=time.perf_counter() - start_time,
        recent_satisfaction=_recent_mean(episode_satisfaction),
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    env_config: ResourceLogisticsConfig,
    episodes: int,
    seed: int,
    device: str = "cpu",
    replay_dir: Path | None = None,
) -> EvaluationResult:
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    env_observation_dim = env_config.observation_dim
    actor = IPPOActor(observation_dim=env_observation_dim, action_dim=env_config.action_dim).to(
        torch_device
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    num_agents = env_config.num_agents
    satisfaction_rates: list[float] = []
    rewards: list[float] = []
    for episode in range(episodes):
        replay_path = (
            replay_dir / f"eval_seed_{seed + episode}.msgpack" if replay_dir is not None else None
        )
        episode_config = replace(env_config, replay_path=replay_path)
        env = ResourceLogisticsEnv(config=episode_config, seed=seed + episode)
        observations, infos = env.reset(seed=seed + episode)
        hidden = actor.initial_hidden(num_agents, torch_device)
        done = False
        while not done:
            obs_tensor = _stack_single_env_observations(observations, env, torch_device)
            mask_tensor = _stack_single_env_action_masks(infos, env, torch_device)
            with torch.no_grad():
                logits, hidden = actor.forward_step(obs_tensor, hidden, mask_tensor)
                actions = torch.argmax(logits, dim=-1)
            action_dict = {
                agent: int(actions[index].item()) for index, agent in enumerate(env.possible_agents)
            }
            observations, _step_rewards, terminations, truncations, infos = env.step(action_dict)
            done = all(terminations.values()) or all(truncations.values())
        outcomes = env.outcomes()
        satisfaction_rates.append(float(cast(int | float, outcomes["demand_satisfaction_rate"])))
        rewards.append(float(cast(int | float, outcomes["total_reward"])))
        env.close()
    return EvaluationResult(
        checkpoint_path=checkpoint_path,
        episodes=episodes,
        satisfaction_rates=satisfaction_rates,
        rewards=rewards,
    )


def bootstrap_success_ci(
    satisfaction_rates: Sequence[float],
    *,
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    if len(satisfaction_rates) == 0:
        raise ValueError("Cannot bootstrap an empty satisfaction-rate sequence")
    rng = np.random.default_rng(seed)
    values = np.asarray(satisfaction_rates, dtype=np.float64)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for sample_id in range(resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[sample_id] = float(np.mean(sample))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        mean=float(np.mean(values)),
        low=float(np.quantile(bootstrap_means, alpha)),
        high=float(np.quantile(bootstrap_means, 1.0 - alpha)),
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages


def _stack_observations(
    observations: Sequence[Mapping[str, np.ndarray]],
    device: torch.device,
) -> torch.Tensor:
    arrays = [
        env_observations[agent]
        for env_observations in observations
        for agent in sorted(env_observations.keys())
    ]
    return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)


def _stack_action_masks(
    infos: Sequence[Mapping[str, Mapping[str, Any]]],
    device: torch.device,
) -> torch.Tensor:
    arrays = [
        env_infos[agent]["action_mask"] for env_infos in infos for agent in sorted(env_infos.keys())
    ]
    return torch.as_tensor(np.stack(arrays), dtype=torch.bool, device=device)


def _stack_single_env_observations(
    observations: Mapping[str, np.ndarray],
    env: ResourceLogisticsEnv,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([observations[agent] for agent in env.possible_agents]),
        dtype=torch.float32,
        device=device,
    )


def _stack_single_env_action_masks(
    infos: Mapping[str, Mapping[str, Any]],
    env: ResourceLogisticsEnv,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([infos[agent]["action_mask"] for agent in env.possible_agents]),
        dtype=torch.bool,
        device=device,
    )


def _unflatten_actions(
    actions: np.ndarray, envs: Sequence[ResourceLogisticsEnv]
) -> list[dict[str, int]]:
    env_actions: list[dict[str, int]] = []
    offset = 0
    for env in envs:
        env_action: dict[str, int] = {}
        for agent in env.possible_agents:
            env_action[agent] = int(actions[offset])
            offset += 1
        env_actions.append(env_action)
    return env_actions


def _save_checkpoint(
    *,
    actor: IPPOActor,
    critic: IPPOCritic,
    env_config: ResourceLogisticsConfig,
    train_config: IPPOTrainConfig,
    global_step: int,
) -> Path:
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = train_config.checkpoint_name or f"ippo_seed_{train_config.seed}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "env_config": env_config.to_dict(),
            "train_config": asdict(train_config),
            "global_step": global_step,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _start_wandb(train_config: IPPOTrainConfig, env_config: ResourceLogisticsConfig) -> Any | None:
    if not train_config.track_wandb:
        return None
    wandb = __import__("wandb")
    return wandb.init(
        project=train_config.wandb_project,
        group=train_config.wandb_group,
        name=f"ippo_seed_{train_config.seed}",
        config={
            "train": asdict(train_config),
            "env": env_config.to_dict(),
        },
        mode=train_config.wandb_mode,
    )


def _recent_mean(values: Sequence[float], window: int = 100) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))
