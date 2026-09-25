from __future__ import annotations

import json

import pytest

from rs_hmcomm.control import LearnedRoutingPolicy
from rs_hmcomm.rl import (
    EvoRSAgentLoop,
    VerlRoutingAdapter,
    verl_compute_score,
    verl_environment_status,
)
from rs_hmcomm.rl.fixtures import Phase5FixtureRunner, build_phase5_fixture_splits


def test_verl_record_and_reward_use_executed_causal_rollout(tmp_path) -> None:
    train, _ = build_phase5_fixture_splits()
    sample = train[0]
    policy = LearnedRoutingPolicy(seed=3)
    rollout = EvoRSAgentLoop(policy, Phase5FixtureRunner()).run(
        sample,
        episode_id="verl-adapter",
        training=True,
        explore=False,
    )
    record = VerlRoutingAdapter.training_record(sample, rollout)
    score = verl_compute_score(
        record["data_source"],
        "model text is not trusted as reward",
        record["reward_model"]["ground_truth"],
        record["extra_info"],
    )

    assert score == rollout.reward["total"]
    assert record["agent_name"] == "evors_routing_agent_loop"
    assert record["extra_info"]["evors_rollout"]["causality"][
        "all_actions_precede_transitions"
    ] is True

    path = VerlRoutingAdapter.write_jsonl(
        (sample,), tmp_path / "verl.jsonl", rollouts={sample.sample_id: rollout}
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["extra_info"]["schema_version"] == "evors.verl-routing-record.v1"


def test_verl_reward_rejects_unaudited_text_only_output() -> None:
    with pytest.raises(ValueError, match="executed"):
        verl_compute_score(
            "evors_phase5_routing",
            '{"reward": 999}',
            "answer",
            {"sample_id": "x"},
        )


def test_verl_status_is_explicit() -> None:
    status = verl_environment_status()
    assert isinstance(status["installed"], bool)
    assert set(status) == {"installed", "version", "error_type"}

