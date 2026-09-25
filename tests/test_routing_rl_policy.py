from __future__ import annotations

import json

import pytest

from rs_hmcomm.control import (
    ACTION_IDS,
    ControllerObservation,
    ControllerWorkspaceSummary,
    DispatchDecision,
    LearnedRoutingPolicy,
    valid_action_mask,
)
from rs_hmcomm.rl.fixtures import Phase5FixtureRunner, build_phase5_fixture_splits
from rs_hmcomm.workspace import WorkspaceState


def _initial_observation(sample):
    workspace = WorkspaceState.initial("policy-test", sample.task)
    return ControllerObservation(
        task=sample.task,
        workspace=ControllerWorkspaceSummary.from_workspace(workspace),
        agent_path=(),
        step=0,
        max_steps=6,
    )


def _biased_policy(action_id: str) -> LearnedRoutingPolicy:
    biases = [0.0] * len(ACTION_IDS)
    biases[ACTION_IDS.index(action_id)] = 8.0
    return LearnedRoutingPolicy(biases=biases, trained_steps=1, seed=3)


def test_untrained_policy_uses_safe_rule_fallback() -> None:
    train, _ = build_phase5_fixture_splits()
    policy = LearnedRoutingPolicy(seed=1)
    decision = policy.decide(_initial_observation(train[0]))
    assert isinstance(decision, DispatchDecision)
    assert decision.expert_id == "visual"
    assert decision.policy_action_id is None


def test_initial_mask_only_allows_visual() -> None:
    train, _ = build_phase5_fixture_splits()
    mask = valid_action_mask(_initial_observation(train[0]))
    assert [ACTION_IDS[index] for index, allowed in enumerate(mask) if allowed] == [
        "dispatch_visual"
    ]


def test_forced_policy_action_changes_real_execution_path_and_cost() -> None:
    train, _ = build_phase5_fixture_splits()
    sample = next(item for item in train if item.task.task_type == "object_quantity")
    runner = Phase5FixtureRunner()

    skip = _biased_policy("dispatch_reasoning_text_structure")
    skip.begin_episode(training=False, explore=False)
    skip_result = runner.run(skip, sample, episode_id="force-skip")

    include = _biased_policy("dispatch_knowledge")
    include.begin_episode(training=False, explore=False)
    include_result = runner.run(include, sample, episode_id="force-include")

    assert skip_result.agent_path == ("visual", "reasoning", "verification")
    assert include_result.agent_path == (
        "visual",
        "knowledge",
        "reasoning",
        "verification",
    )
    assert skip_result.workspace.version == 3
    assert include_result.workspace.version == 4
    assert skip_result.total_cost.agent_calls == 3
    assert include_result.total_cost.agent_calls == 4
    assert [item["policy_action_id"] for item in (
        transition.decision for transition in skip_result.transitions
    )] == [sample.action_id for sample in skip.episode_trace]


def test_policy_gradient_update_and_checkpoint_round_trip(tmp_path) -> None:
    train, _ = build_phase5_fixture_splits()
    policy = LearnedRoutingPolicy(seed=7)
    policy.begin_episode(training=True, explore=False)
    result = Phase5FixtureRunner().run(policy, train[0], episode_id="update")
    result.replay()
    before = policy.parameter_digest()
    update = policy.update_episode(0.75, learning_rate=0.1)
    after = policy.parameter_digest()

    assert before != after
    assert update["update_l2"] > 0
    path = policy.save(tmp_path / "policy.json")
    loaded = LearnedRoutingPolicy.from_checkpoint(path)
    assert loaded.parameter_digest() == policy.parameter_digest()
    assert loaded.checkpoint_dict() == policy.checkpoint_dict()

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["biases"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        LearnedRoutingPolicy.from_checkpoint(tampered)


def test_reference_answer_is_environment_only() -> None:
    train, _ = build_phase5_fixture_splits()
    for sample in train:
        assert sample.reference_answer not in json.dumps(
            sample.task.to_dict(), ensure_ascii=False, sort_keys=True
        )

