from __future__ import annotations

import os
from pathlib import Path

import pytest
from babel.algorithms.ippo import bootstrap_success_ci, evaluate_checkpoint
from babel.env.dynamics import ResourceLogisticsConfig


def test_bootstrap_success_ci_is_deterministic_for_fixed_seed() -> None:
    rates = [0.2, 0.4, 0.6, 0.8]

    first = bootstrap_success_ci(rates, seed=11, resamples=200)
    second = bootstrap_success_ci(rates, seed=11, resamples=200)

    assert first == second
    assert first.mean == pytest.approx(0.5)
    assert first.low <= first.mean <= first.high


def test_fast_calibration_eval_smoke_runs_under_100_episodes(tmp_path) -> None:
    checkpoint = os.environ.get("BABEL_IPPO_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set BABEL_IPPO_CHECKPOINT to run checkpoint calibration smoke.")

    result = evaluate_checkpoint(
        checkpoint_path=Path(checkpoint),
        env_config=ResourceLogisticsConfig(replay_path=None),
        episodes=16,
        seed=90_000,
        device="cpu",
        replay_dir=tmp_path,
    )

    assert result.episodes == 16
    assert 0.0 <= result.mean_satisfaction <= 1.0
    assert len(result.satisfaction_rates) == 16


@pytest.mark.slow
def test_full_500_episode_calibration_gate() -> None:
    checkpoint = os.environ.get("BABEL_IPPO_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set BABEL_IPPO_CHECKPOINT to run the full calibration gate.")

    result = evaluate_checkpoint(
        checkpoint_path=Path(checkpoint),
        env_config=ResourceLogisticsConfig(replay_path=None),
        episodes=500,
        seed=100_000,
        device="cpu",
    )

    ci = bootstrap_success_ci(result.satisfaction_rates, seed=19, resamples=1000)
    assert 0.25 <= ci.mean <= 0.45
