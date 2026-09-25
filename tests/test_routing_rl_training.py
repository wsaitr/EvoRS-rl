from __future__ import annotations

from rs_hmcomm.control import LearnedRoutingPolicy
from rs_hmcomm.rl import (
    RoutingPolicyTrainer,
    RoutingRewardBreakdown,
    RoutingRewardManager,
    RoutingTrainerConfig,
    audit_action_causality,
)
from rs_hmcomm.rl.fixtures import Phase5FixtureRunner, build_phase5_fixture_splits


def test_reward_components_recompute_exactly() -> None:
    train, _ = build_phase5_fixture_splits()
    policy = LearnedRoutingPolicy(seed=1)
    policy.begin_episode(training=True, explore=False)
    result = Phase5FixtureRunner().run(policy, train[0], episode_id="reward")
    reward = RoutingRewardManager().score(result, reference_answer=train[0].reference_answer)

    assert isinstance(reward, RoutingRewardBreakdown)
    assert reward.task_score == 1.0
    assert reward.agent_cost > 0
    assert reward.total == (
        reward.task_reward
        + reward.evidence_quality_gain
        + reward.structure_quality_gain
        - reward.message_cost
        - reward.agent_cost
        - reward.tool_cost
        - reward.model_cost
        - reward.redundancy_cost
        - reward.invalid_action_cost
        - reward.error_cost
        - reward.non_completion_cost
    )


def test_sampled_actions_are_causally_linked_to_pre_state() -> None:
    train, _ = build_phase5_fixture_splits()
    policy = LearnedRoutingPolicy(seed=2)
    policy.begin_episode(training=True, explore=False)
    result = Phase5FixtureRunner().run(policy, train[0], episode_id="causal")
    audit = audit_action_causality(policy.episode_trace, result)

    assert audit["all_actions_precede_transitions"] is True
    assert audit["linked_action_count"] == len(result.transitions)
    for link in audit["links"]:
        assert link["workspace_version_at_sample"] == link[
            "transition_workspace_version_before"
        ]


def test_trainer_updates_only_policy_and_eval_is_frozen() -> None:
    train, evaluate = build_phase5_fixture_splits()
    small_train = (train[0], train[6])
    policy = LearnedRoutingPolicy(seed=11)
    result = RoutingPolicyTrainer(
        policy,
        Phase5FixtureRunner(),
        config=RoutingTrainerConfig(epochs=2, learning_rate=0.1, seed=11),
    ).train_and_evaluate(small_train, evaluate[:2])

    assert result.initial_parameter_sha256 != result.final_parameter_sha256
    assert result.experts_frozen is True
    assert all(event["optimizer_update"]["update_l2"] > 0 for event in result.train_events)
    assert all(
        event["parameter_sha256_before"] == event["parameter_sha256_after"]
        for event in result.eval_events
    )
    assert all(
        event["causality"]["all_actions_precede_transitions"]
        for event in (*result.train_events, *result.eval_events)
    )

