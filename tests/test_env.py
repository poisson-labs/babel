from __future__ import annotations

import msgpack
import numpy as np
from babel.env.dynamics import (
    NUM_AGENTS,
    ResourceLogisticsConfig,
    ResourceLogisticsEnv,
)
from babel.env.graph import DEMAND, DEPOT, NUM_NODES, TRANSIT, generate_resource_graph
from babel.env.replay import load_replay


def test_graph_generation_is_deterministic_and_matches_topology() -> None:
    graph_a = generate_resource_graph(seed=17)
    graph_b = generate_resource_graph(seed=17)

    assert graph_a.to_dict() == graph_b.to_dict()
    assert len(graph_a.node_types) == NUM_NODES
    assert graph_a.node_types.count(DEPOT) == 3
    assert graph_a.node_types.count(DEMAND) == 6
    assert graph_a.node_types.count(TRANSIT) == 3
    assert all(2 <= len(graph_a.neighbors(node_id)) <= 4 for node_id in range(NUM_NODES))
    assert all(edge.cost in {1, 2, 3} for edge in graph_a.edges)


def test_graph_connectivity_invariant_holds_across_seed_range() -> None:
    for seed in range(50):
        graph = generate_resource_graph(seed=seed)
        assert graph.connectivity_invariant_holds(max_hops=4)


def test_env_reset_step_and_action_masks_are_well_formed() -> None:
    env = ResourceLogisticsEnv(ResourceLogisticsConfig(), seed=123)
    observations, infos = env.reset(seed=123)

    assert len(observations) == NUM_AGENTS
    assert set(observations) == set(env.agents)
    assert all(obs.shape == (env.config.observation_dim,) for obs in observations.values())
    assert all(np.isfinite(obs).all() for obs in observations.values())
    assert all("action_mask" in info for info in infos.values())

    global_state = env.global_state_features()
    assert global_state.shape == (env.config.global_state_dim,)
    assert np.isfinite(global_state).all()

    action_masks = env.action_masks()
    for agent_id, mask in action_masks.items():
        assert mask.shape == (env.config.action_dim,)
        assert mask.dtype == np.bool_
        assert mask[env.noop_action]
        assert np.any(mask)
        assert np.array_equal(mask, infos[agent_id]["action_mask"])

    actions = {agent_id: env.sample_valid_action(agent_id) for agent_id in env.agents}
    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

    assert set(next_obs) == set(env.agents)
    assert all(obs.shape == (env.config.observation_dim,) for obs in next_obs.values())
    assert len(set(rewards.values())) == 1
    assert all(isinstance(done, bool) for done in terminations.values())
    assert all(isinstance(done, bool) for done in truncations.values())
    assert all("action_mask" in info for info in next_infos.values())


def test_replay_round_trip_records_exhaustive_episode(tmp_path) -> None:
    replay_path = tmp_path / "episode.msgpack"
    env = ResourceLogisticsEnv(ResourceLogisticsConfig(replay_path=replay_path), seed=321)
    env.reset(seed=321)

    done = False
    while not done:
        actions = {agent_id: env.sample_valid_action(agent_id) for agent_id in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())

    env.close()

    replay = load_replay(replay_path)
    assert replay["seed"] == 321
    assert replay["config"]["num_nodes"] == NUM_NODES
    assert len(replay["per_step"]) > 0
    assert set(replay["outcomes"]) >= {"demands_satisfied", "demands_missed", "total_reward"}

    first_step = replay["per_step"][0]
    assert set(first_step) >= {"step", "env_state", "per_agent"}
    assert len(first_step["per_agent"]) == NUM_AGENTS
    assert set(first_step["per_agent"][0]) >= {
        "obs",
        "action",
        "message_sent_raw",
        "message_sent_perturbed",
        "messages_received",
        "reward",
    }

    with replay_path.open("rb") as handle:
        packed = msgpack.unpackb(handle.read(), raw=False)
    assert packed == replay


def test_message_history_component_is_zero_until_messages_arrive() -> None:
    env = ResourceLogisticsEnv(ResourceLogisticsConfig(message_dim=16), seed=222)
    observations, _ = env.reset(seed=222)
    message_history_offset = env.config.message_history_offset
    message_end = message_history_offset + env.config.message_history_length * 3 * 16

    assert all(obs.shape == (env.config.observation_dim,) for obs in observations.values())
    assert all(
        np.allclose(obs[message_history_offset:message_end], 0.0) for obs in observations.values()
    )

    sender = env.possible_agents[0]
    receiver = env.possible_agents[1]
    embedding = np.ones(16, dtype=np.float32)
    actions = {agent: env.noop_action for agent in env.possible_agents}
    actions[sender] = {
        "movement": env.noop_action,
        "message_embeddings": {receiver: embedding},
        "message_raw": {"slots": [1, 2, 1]},
    }

    next_observations, _, _, _, _ = env.step(actions)
    receiver_history = next_observations[receiver][message_history_offset:message_end]

    assert np.count_nonzero(receiver_history) == 16
