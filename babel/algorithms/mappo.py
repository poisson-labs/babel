from __future__ import annotations

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from babel.algorithms.ippo import BootstrapCI, EvaluationResult, bootstrap_success_ci
from babel.channels.symbolic import SymbolicChannel
from babel.env.dynamics import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    GLOBAL_STATE_DIM,
    MESSAGE_HISTORY_OFFSET,
    NUM_AGENTS,
    ResourceLogisticsConfig,
    ResourceLogisticsEnv,
)
from babel.networks.encoders import ObservationEncoder

_torch = cast(Any, torch)


@dataclass(frozen=True, slots=True)
class SymbolicChannelConfig:
    message_dim: int = 16
    hidden_dim: int = 128
    initial_temperature: float = 1.0
    min_temperature: float = 0.2
    anneal_steps: int = 1_000_000


@dataclass(frozen=True, slots=True)
class MAPPOTrainConfig:
    total_timesteps: int = 7_000_000
    num_envs: int = 16
    rollout_length: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    channel_learning_rate: float = 1e-4
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
    wandb_group: str = "phase2_symbolic_8bit"
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
class Gate2Result:
    ci: BootstrapCI
    ippo_point_gap: float
    ippo_upper_ci_gap: float
    status: str
    passed: bool


class MessageAggregator(nn.Module):
    def __init__(self, *, message_dim: int, query_dim: int = 128, context_dim: int = 64) -> None:
        super().__init__()
        self.message_dim = message_dim
        self.context_dim = context_dim
        if message_dim > 0:
            self.query = nn.Linear(query_dim, context_dim)
            self.key = nn.Linear(message_dim, context_dim)
            self.value = nn.Linear(message_dim, context_dim)
        else:
            self.query = nn.Identity()
            self.key = nn.Identity()
            self.value = nn.Identity()

    def forward(self, hidden: Tensor, observations: Tensor) -> Tensor:
        if self.message_dim == 0:
            return _torch.zeros(
                (hidden.shape[0], self.context_dim),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        history_width = observations.shape[1] - BASE_OBSERVATION_DIM
        history_start = MESSAGE_HISTORY_OFFSET
        history_stop = history_start + history_width
        history = observations[:, history_start:history_stop]
        messages = history.reshape(hidden.shape[0], -1, self.message_dim)
        mask = messages.abs().sum(dim=-1) > 0
        query = self.query(hidden).unsqueeze(1)
        keys = self.key(messages)
        values = self.value(messages)
        scores = (query * keys).sum(dim=-1) / float(self.context_dim**0.5)
        scores = scores.masked_fill(~mask, -1.0e9)
        no_messages = ~mask.any(dim=-1)
        attention = torch.softmax(scores, dim=-1)
        attention = attention.masked_fill(no_messages.unsqueeze(-1), 0.0)
        return cast(Tensor, (attention.unsqueeze(-1) * values).sum(dim=1))


class MAPPOActor(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        message_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = ObservationEncoder(
            observation_dim=BASE_OBSERVATION_DIM,
            hidden_dim=hidden_dim,
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.message_aggregator = MessageAggregator(
            message_dim=message_dim,
            query_dim=hidden_dim,
            context_dim=64,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.observation_dim = observation_dim
        self.message_history_width = observation_dim - BASE_OBSERVATION_DIM

    def initial_hidden(self, batch_size: int, device: object) -> Tensor:
        return cast(
            Tensor,
            _torch.zeros((batch_size, self.hidden_dim), dtype=_torch.float32, device=device),
        )

    def forward_step(
        self,
        observations: Tensor,
        hidden: Tensor,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base_observations = self._base_observations(observations)
        encoded = self.encoder(base_observations)
        next_hidden = self.gru(encoded, hidden)
        message_context = self.message_aggregator(next_hidden, observations)
        logits = self.policy_head(_torch.cat([next_hidden, message_context], dim=-1))
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float(_torch.finfo(logits.dtype).min))
        return logits, next_hidden, next_hidden

    def _base_observations(self, observations: Tensor) -> Tensor:
        if self.message_history_width == 0:
            return observations[:, :BASE_OBSERVATION_DIM]
        history_stop = MESSAGE_HISTORY_OFFSET + self.message_history_width
        return _torch.cat(
            [observations[:, :MESSAGE_HISTORY_OFFSET], observations[:, history_stop:]],
            dim=-1,
        )


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        global_state_dim: int = GLOBAL_STATE_DIM,
        num_agents: int = NUM_AGENTS,
    ) -> None:
        super().__init__()
        input_dim = observation_dim * num_agents + global_state_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, all_agent_observations: Tensor) -> Tensor:
        return self.network(all_agent_observations).squeeze(-1)


def train_mappo_symbolic(
    *,
    env_config: ResourceLogisticsConfig,
    train_config: MAPPOTrainConfig,
    channel_config: SymbolicChannelConfig,
) -> TrainingResult:
    _set_global_seeds(train_config.seed)
    device = torch.device(train_config.device)
    env_config = replace(
        env_config,
        message_dim=channel_config.message_dim,
        message_history_length=5,
        replay_path=None,
    )
    observation_dim = env_config.observation_dim
    envs = [
        ResourceLogisticsEnv(config=env_config, seed=train_config.seed + env_id)
        for env_id in range(train_config.num_envs)
    ]
    current_seed = train_config.seed + train_config.num_envs
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, dict[str, Any]]] = []
    for env_id, env in enumerate(envs):
        obs, info = env.reset(seed=train_config.seed + env_id)
        observations.append(obs)
        infos.append(info)

    actor = MAPPOActor(
        observation_dim=observation_dim,
        message_dim=channel_config.message_dim,
    ).to(device)
    critic = CentralizedCritic(observation_dim=observation_dim).to(device)
    channel = SymbolicChannel(
        agent_intent_dim=actor.hidden_dim,
        observation_dim=observation_dim,
        message_dim=channel_config.message_dim,
        hidden_dim=channel_config.hidden_dim,
        initial_temperature=channel_config.initial_temperature,
        min_temperature=channel_config.min_temperature,
        anneal_steps=channel_config.anneal_steps,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": train_config.learning_rate},
            {"params": critic.parameters(), "lr": train_config.learning_rate},
            {"params": channel.parameters(), "lr": train_config.channel_learning_rate},
        ],
        eps=1e-5,
    )
    batch_size = train_config.num_envs * NUM_AGENTS
    rollout_env_batch = train_config.rollout_length * train_config.num_envs
    central_observation_dim = NUM_AGENTS * observation_dim + GLOBAL_STATE_DIM
    rollout_actor_batch = rollout_env_batch * NUM_AGENTS
    minibatch_size = min(train_config.minibatch_size, rollout_actor_batch)
    updates = max(ceil(train_config.total_timesteps / rollout_actor_batch), 1)
    actor_hidden = actor.initial_hidden(batch_size, device)
    run = _start_wandb(train_config, env_config, channel_config)
    global_step = 0
    episode_satisfaction: list[float] = []
    start_time = time.perf_counter()

    for update in range(1, updates + 1):
        obs_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, NUM_AGENTS, observation_dim),
            device=device,
        )
        central_obs_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, central_observation_dim),
            device=device,
        )
        mask_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, NUM_AGENTS, ACTION_DIM),
            dtype=torch.bool,
            device=device,
        )
        action_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, NUM_AGENTS),
            dtype=torch.long,
            device=device,
        )
        message_slot_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, NUM_AGENTS, 3),
            dtype=torch.long,
            device=device,
        )
        logprob_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, NUM_AGENTS),
            device=device,
        )
        reward_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs),
            device=device,
        )
        done_buf = torch.zeros((train_config.rollout_length, train_config.num_envs), device=device)
        value_buf = torch.zeros((train_config.rollout_length, train_config.num_envs), device=device)
        hidden_buf = torch.zeros(
            (train_config.rollout_length, train_config.num_envs, NUM_AGENTS, actor.hidden_dim),
            device=device,
        )

        for step in range(train_config.rollout_length):
            channel.anneal_temperature(global_step)
            obs_tensor = _stack_observations(observations, device)
            flat_obs = obs_tensor.reshape(batch_size, observation_dim)
            central_obs = _central_observations(obs_tensor, envs, device)
            mask_tensor = _stack_action_masks(infos, device)
            flat_mask = mask_tensor.reshape(batch_size, ACTION_DIM)

            obs_buf[step] = obs_tensor
            central_obs_buf[step] = central_obs
            mask_buf[step] = mask_tensor
            hidden_buf[step] = actor_hidden.reshape(
                train_config.num_envs,
                NUM_AGENTS,
                actor.hidden_dim,
            )

            with torch.no_grad():
                logits, next_actor_hidden, agent_intents = actor.forward_step(
                    flat_obs,
                    actor_hidden,
                    flat_mask,
                )
                values = critic(central_obs)
                action_distribution = Categorical(logits=logits)
                movement_actions = action_distribution.sample()
                action_logprobs = action_distribution.log_prob(movement_actions)
                message_batch = channel.sample_batch(agent_intents, flat_obs)
                combined_logprobs = action_logprobs + message_batch.logprobs

            action_buf[step] = movement_actions.reshape(train_config.num_envs, NUM_AGENTS)
            message_slot_buf[step] = message_batch.slots.reshape(
                train_config.num_envs, NUM_AGENTS, 3
            )
            logprob_buf[step] = combined_logprobs.reshape(train_config.num_envs, NUM_AGENTS)
            value_buf[step] = values

            env_actions = _unflatten_symbolic_actions(
                movement_actions.cpu().numpy(),
                message_batch.embeddings.detach().cpu().numpy(),
                message_batch.slots.detach().cpu().numpy(),
                envs,
            )
            next_observations: list[dict[str, np.ndarray]] = []
            next_infos: list[dict[str, dict[str, Any]]] = []
            done_flags = np.zeros(train_config.num_envs, dtype=np.float32)

            for env_id, env in enumerate(envs):
                obs, rewards, terminations, truncations, info = env.step(env_actions[env_id])
                done = all(terminations.values()) or all(truncations.values())
                reward_buf[step, env_id] = float(rewards[env.possible_agents[0]])
                if done:
                    done_flags[env_id] = 1.0
                    outcome = cast(
                        dict[str, Any], next(iter(info.values())).get("outcomes", env.outcomes())
                    )
                    episode_satisfaction.append(float(outcome["demand_satisfaction_rate"]))
                    obs, info = env.reset(seed=current_seed)
                    current_seed += 1
                    start = env_id * NUM_AGENTS
                    stop = start + NUM_AGENTS
                    next_actor_hidden[start:stop] = 0.0
                next_observations.append(obs)
                next_infos.append(info)

            done_buf[step] = torch.as_tensor(done_flags, dtype=torch.float32, device=device)
            observations = next_observations
            infos = next_infos
            actor_hidden = next_actor_hidden.detach()
            global_step += batch_size

        with torch.no_grad():
            next_obs_tensor = _stack_observations(observations, device)
            next_central_obs = _central_observations(next_obs_tensor, envs, device)
            next_values = critic(next_central_obs)
            advantages = _compute_gae(
                rewards=reward_buf,
                dones=done_buf,
                values=value_buf,
                next_value=next_values,
                gamma=train_config.gamma,
                gae_lambda=train_config.gae_lambda,
            )
            returns = advantages + value_buf

        actor_obs = obs_buf.reshape(rollout_actor_batch, observation_dim)
        actor_masks = mask_buf.reshape(rollout_actor_batch, ACTION_DIM)
        actor_actions = action_buf.reshape(rollout_actor_batch)
        actor_slots = message_slot_buf.reshape(rollout_actor_batch, 3)
        old_logprobs = logprob_buf.reshape(rollout_actor_batch)
        actor_hidden_flat = hidden_buf.reshape(rollout_actor_batch, actor.hidden_dim)
        actor_advantages = (
            advantages.unsqueeze(-1).expand(-1, -1, NUM_AGENTS).reshape(rollout_actor_batch)
        )
        actor_advantages = (actor_advantages - actor_advantages.mean()) / (
            actor_advantages.std() + 1e-8
        )

        critic_obs = central_obs_buf.reshape(rollout_env_batch, central_observation_dim)
        critic_returns = returns.reshape(rollout_env_batch)
        flat_values = value_buf.reshape(rollout_env_batch)

        actor_indices = np.arange(rollout_actor_batch)
        critic_indices = np.arange(rollout_env_batch)
        last_metrics: dict[str, float] = {}
        for _epoch in range(train_config.ppo_epochs):
            np.random.shuffle(actor_indices)
            np.random.shuffle(critic_indices)
            for start in range(0, rollout_actor_batch, minibatch_size):
                mb_indices = torch.as_tensor(
                    actor_indices[start : start + minibatch_size],
                    dtype=torch.long,
                    device=device,
                )
                logits, _hidden, agent_intents = actor.forward_step(
                    actor_obs[mb_indices],
                    actor_hidden_flat[mb_indices],
                    actor_masks[mb_indices],
                )
                action_distribution = Categorical(logits=logits)
                action_logprobs = action_distribution.log_prob(actor_actions[mb_indices])
                action_entropy = action_distribution.entropy()
                message_logprobs, message_entropy = channel.evaluate_slots(
                    agent_intents,
                    actor_obs[mb_indices],
                    actor_slots[mb_indices],
                )
                new_logprobs = action_logprobs + message_logprobs
                entropy = (action_entropy + message_entropy).mean()
                logratio = new_logprobs - old_logprobs[mb_indices]
                ratio = logratio.exp()
                pg_loss_1 = -actor_advantages[mb_indices] * ratio
                pg_loss_2 = -actor_advantages[mb_indices] * torch.clamp(
                    ratio,
                    1.0 - train_config.clip_coef,
                    1.0 + train_config.clip_coef,
                )
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                critic_start = start % rollout_env_batch
                critic_stop = critic_start + min(minibatch_size, rollout_env_batch)
                critic_mb = torch.as_tensor(
                    critic_indices[critic_start:critic_stop],
                    dtype=torch.long,
                    device=device,
                )
                if critic_mb.numel() == 0:
                    critic_mb = torch.as_tensor(
                        critic_indices[:minibatch_size],
                        dtype=torch.long,
                        device=device,
                    )
                new_values = critic(critic_obs[critic_mb])
                value_loss = 0.5 * ((new_values - critic_returns[critic_mb]) ** 2).mean()
                loss = (
                    pg_loss
                    + train_config.value_coef * value_loss
                    - train_config.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                parameters = (
                    list(actor.parameters())
                    + list(critic.parameters())
                    + list(channel.parameters())
                )
                nn.utils.clip_grad_norm_(
                    parameters,
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
                    "charts/channel_temperature": float(channel.temperature),
                    "charts/global_step": float(global_step),
                    "charts/recent_satisfaction": _recent_mean(episode_satisfaction),
                    "charts/value_mean": float(flat_values.mean().item()),
                }

        if run is not None and (update % train_config.log_interval_updates == 0 or update == 1):
            run.log(last_metrics, step=global_step)

    checkpoint_path = _save_checkpoint(
        actor=actor,
        critic=critic,
        channel=channel,
        env_config=env_config,
        train_config=train_config,
        channel_config=channel_config,
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


def evaluate_symbolic_checkpoint(
    *,
    checkpoint_path: Path,
    env_config: ResourceLogisticsConfig,
    episodes: int,
    seed: int,
    device: str = "cpu",
) -> EvaluationResult:
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    channel_config = SymbolicChannelConfig(**checkpoint["channel_config"])
    env_config = replace(
        env_config,
        message_dim=channel_config.message_dim,
        message_history_length=5,
        replay_path=None,
    )
    observation_dim = env_config.observation_dim
    actor = MAPPOActor(
        observation_dim=observation_dim,
        message_dim=channel_config.message_dim,
    ).to(torch_device)
    channel = SymbolicChannel(
        agent_intent_dim=actor.hidden_dim,
        observation_dim=observation_dim,
        message_dim=channel_config.message_dim,
        hidden_dim=channel_config.hidden_dim,
        initial_temperature=channel_config.initial_temperature,
        min_temperature=channel_config.min_temperature,
        anneal_steps=channel_config.anneal_steps,
    ).to(torch_device)
    actor.load_state_dict(checkpoint["actor"])
    channel.load_state_dict(checkpoint["channel"])
    actor.eval()
    channel.eval()

    satisfaction_rates: list[float] = []
    rewards: list[float] = []
    for episode in range(episodes):
        env = ResourceLogisticsEnv(config=env_config, seed=seed + episode)
        observations, infos = env.reset(seed=seed + episode)
        hidden = actor.initial_hidden(NUM_AGENTS, torch_device)
        done = False
        while not done:
            obs_tensor = _stack_single_env_observations(observations, env, torch_device)
            mask_tensor = _stack_single_env_action_masks(infos, env, torch_device)
            with torch.no_grad():
                logits, hidden, agent_intents = actor.forward_step(obs_tensor, hidden, mask_tensor)
                movement_actions = torch.argmax(logits, dim=-1)
                message_batch = channel.deterministic_batch(agent_intents, obs_tensor)
            action_dict = _single_env_symbolic_actions(
                movement_actions.cpu().numpy(),
                message_batch.embeddings.cpu().numpy(),
                message_batch.slots.cpu().numpy(),
                env,
            )
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


def evaluate_gate2(
    satisfaction_rates: Sequence[float],
    *,
    seed: int,
    resamples: int,
    ippo_point_estimate: float = 0.34297380952380957,
    ippo_upper_ci: float = 0.3578999404761905,
) -> Gate2Result:
    ci = bootstrap_success_ci(satisfaction_rates, seed=seed, resamples=resamples)
    point_gap = ci.mean - ippo_point_estimate
    conservative_gap = ci.mean - ippo_upper_ci
    if point_gap < 0.20:
        status = "fail_comm_gap_lt_20pp"
        passed = False
    elif point_gap < 0.30:
        status = "borderline_gap_20_to_30pp"
        passed = False
    elif 0.643 <= ci.mean < 0.70:
        status = "borderline_pass_gap_ok_below_band"
        passed = False
    elif 0.70 <= ci.mean <= 0.85:
        status = "passed"
        passed = True
    else:
        status = "fail_outside_target_band"
        passed = False
    return Gate2Result(
        ci=ci,
        ippo_point_gap=point_gap,
        ippo_upper_ci_gap=conservative_gap,
        status=status,
        passed=passed,
    )


def _compute_gae(
    *,
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    next_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
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
) -> Tensor:
    arrays = [
        [env_observations[f"agent_{agent_id}"] for agent_id in range(NUM_AGENTS)]
        for env_observations in observations
    ]
    return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)


def _central_observations(
    observations: Tensor,
    envs: Sequence[ResourceLogisticsEnv],
    device: torch.device,
) -> Tensor:
    agent_observations = observations.reshape(observations.shape[0], -1)
    global_features = torch.as_tensor(
        np.stack([env.global_state_features() for env in envs]),
        dtype=torch.float32,
        device=device,
    )
    return cast(Tensor, _torch.cat([agent_observations, global_features], dim=-1))


def _stack_action_masks(
    infos: Sequence[Mapping[str, Mapping[str, Any]]],
    device: torch.device,
) -> Tensor:
    arrays = [
        [env_infos[f"agent_{agent_id}"]["action_mask"] for agent_id in range(NUM_AGENTS)]
        for env_infos in infos
    ]
    return torch.as_tensor(np.stack(arrays), dtype=torch.bool, device=device)


def _stack_single_env_observations(
    observations: Mapping[str, np.ndarray],
    env: ResourceLogisticsEnv,
    device: torch.device,
) -> Tensor:
    return torch.as_tensor(
        np.stack([observations[agent] for agent in env.possible_agents]),
        dtype=torch.float32,
        device=device,
    )


def _stack_single_env_action_masks(
    infos: Mapping[str, Mapping[str, Any]],
    env: ResourceLogisticsEnv,
    device: torch.device,
) -> Tensor:
    return torch.as_tensor(
        np.stack([infos[agent]["action_mask"] for agent in env.possible_agents]),
        dtype=torch.bool,
        device=device,
    )


def _unflatten_symbolic_actions(
    movement_actions: np.ndarray,
    message_embeddings: np.ndarray,
    message_slots: np.ndarray,
    envs: Sequence[ResourceLogisticsEnv],
) -> list[dict[str, dict[str, object]]]:
    env_actions: list[dict[str, dict[str, object]]] = []
    offset = 0
    for env in envs:
        env_action: dict[str, dict[str, object]] = {}
        for agent in env.possible_agents:
            embedding = message_embeddings[offset]
            receivers = [receiver for receiver in env.possible_agents if receiver != agent]
            env_action[agent] = {
                "movement": int(movement_actions[offset]),
                "message_embeddings": {receiver: embedding.tolist() for receiver in receivers},
                "message_raw": {"slots": message_slots[offset].astype(int).tolist()},
            }
            offset += 1
        env_actions.append(env_action)
    return env_actions


def _single_env_symbolic_actions(
    movement_actions: np.ndarray,
    message_embeddings: np.ndarray,
    message_slots: np.ndarray,
    env: ResourceLogisticsEnv,
) -> dict[str, dict[str, object]]:
    env_action: dict[str, dict[str, object]] = {}
    for agent_index, agent in enumerate(env.possible_agents):
        receivers = [receiver for receiver in env.possible_agents if receiver != agent]
        env_action[agent] = {
            "movement": int(movement_actions[agent_index]),
            "message_embeddings": {
                receiver: message_embeddings[agent_index].tolist() for receiver in receivers
            },
            "message_raw": {"slots": message_slots[agent_index].astype(int).tolist()},
        }
    return env_action


def _save_checkpoint(
    *,
    actor: MAPPOActor,
    critic: CentralizedCritic,
    channel: SymbolicChannel,
    env_config: ResourceLogisticsConfig,
    train_config: MAPPOTrainConfig,
    channel_config: SymbolicChannelConfig,
    global_step: int,
) -> Path:
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = train_config.checkpoint_name or f"mappo_symbolic_seed_{train_config.seed}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "channel": channel.state_dict(),
            "env_config": env_config.to_dict(),
            "train_config": asdict(train_config),
            "channel_config": asdict(channel_config),
            "global_step": global_step,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _start_wandb(
    train_config: MAPPOTrainConfig,
    env_config: ResourceLogisticsConfig,
    channel_config: SymbolicChannelConfig,
) -> Any | None:
    if not train_config.track_wandb:
        return None
    wandb = __import__("wandb")
    return wandb.init(
        project=train_config.wandb_project,
        group=train_config.wandb_group,
        name=f"mappo_symbolic_seed_{train_config.seed}",
        config={
            "train": asdict(train_config),
            "env": env_config.to_dict(),
            "channel": asdict(channel_config),
        },
        mode=train_config.wandb_mode,
    )


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _recent_mean(values: Sequence[float], window: int = 100) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))
