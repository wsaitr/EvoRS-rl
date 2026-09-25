from __future__ import annotations

import json

import pytest

from rs_hmcomm.contracts.runtime import (
    BudgetExceeded,
    BudgetLimits,
    CostVector,
    EpisodeEvent,
    EpisodeTrace,
    EventType,
)


def make_event(seq: int, *, episode_id: str = "episode-1") -> EpisodeEvent:
    return EpisodeEvent(
        episode_id=episode_id,
        event_id=f"event-{seq}",
        seq=seq,
        event_type=EventType.AGENT_DISPATCHED,
        actor="generalist",
        action={"agent_id": "generalist", "assignment_id": f"assignment-{seq}"},
        result={"status": "ok"},
        cost_delta=CostVector(agent_calls=1),
        elapsed_ms=float(seq),
    )


def test_cost_vector_is_additive_and_round_trips() -> None:
    left = CostVector(agent_calls=1, message_structure_bytes=8, billed_compute_ms=1.5)
    right = CostVector(tool_calls=2, message_latent_bytes=32, billed_compute_ms=2.5)
    total = left + right

    assert total.agent_calls == 1
    assert total.tool_calls == 2
    assert total.communication_bytes == 40
    assert total.billed_compute_ms == 4.0
    assert CostVector.from_dict(total.to_dict()) == total


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_calls": -1},
        {"tool_calls": True},
        {"billed_compute_ms": float("inf")},
    ],
)
def test_cost_vector_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        CostVector(**kwargs)


def test_budget_checks_projected_usage_by_dimension() -> None:
    budget = BudgetLimits(
        max_agent_calls=2,
        max_tool_calls=1,
        max_billed_compute_ms=10,
        max_wall_time_ms=20,
    )
    usage = CostVector(agent_calls=2, billed_compute_ms=4)
    delta = CostVector(tool_calls=2)

    assert budget.allows(usage)
    assert not budget.allows(usage, delta)
    with pytest.raises(BudgetExceeded) as exc_info:
        budget.require_within(usage, delta)
    assert exc_info.value.dimensions == ("tool_calls",)
    assert not budget.allows(usage, wall_time_ms=21)


def test_trace_append_gates_the_actual_event_cost_delta() -> None:
    trace = EpisodeTrace(
        "episode-1",
        budget_limits=BudgetLimits(max_agent_calls=0),
    )
    with pytest.raises(BudgetExceeded, match="budget exceeded"):
        trace.append(make_event(0))


def test_event_and_trace_json_replay_preserve_cost() -> None:
    trace = EpisodeTrace("episode-1").append(make_event(0)).append(make_event(1))
    decoded = EpisodeEvent.from_json(trace.events[0].to_json())
    replayed = EpisodeTrace.replay(EpisodeEvent.from_dict(item) for item in trace.to_dict()["events"])

    assert decoded == trace.events[0]
    assert replayed == trace
    assert replayed.total_cost.agent_calls == 2
    assert replayed.ledger.total == replayed.total_cost
    assert json.loads(trace.events[0].to_json())["event_type"] == "agent_dispatched"


def test_trace_rejects_gaps_duplicates_and_cross_episode_events() -> None:
    trace = EpisodeTrace("episode-1").append(make_event(0))
    with pytest.raises(ValueError, match="next event seq"):
        trace.append(make_event(2))
    with pytest.raises(ValueError, match="episode_id"):
        trace.append(make_event(1, episode_id="other"))
    duplicate_id = EpisodeEvent(
        episode_id="episode-1",
        event_id="event-0",
        seq=1,
        event_type=EventType.WARNING,
        actor="runtime",
        action={},
        result={},
        elapsed_ms=1.0,
    )
    with pytest.raises(ValueError, match="duplicate event_id"):
        trace.append(duplicate_id)


def test_event_rejects_non_json_runtime_objects() -> None:
    with pytest.raises((TypeError, ValueError), match="unsupported"):
        EpisodeEvent(
            episode_id="episode-1",
            event_id="event-0",
            seq=0,
            event_type=EventType.OBSERVATION_BUILT,
            actor="runtime",
            action={"image": object()},
            result={},
        )


def test_event_cost_has_one_owner() -> None:
    with pytest.raises(ValueError, match="cannot own cost fields"):
        EpisodeEvent(
            episode_id="episode-1",
            event_id="event-cost",
            seq=0,
            event_type=EventType.AGENT_ACTED,
            actor="generalist",
            action={},
            result={},
            cost_delta=CostVector(model_calls=1),
        )


@pytest.mark.parametrize(
    ("event_type", "action", "result", "cost"),
    [
        (
            EventType.AGENT_DISPATCHED,
            {"agent_id": "generalist", "assignment_id": "assignment-0", "call_id": "dispatch-0"},
            {},
            CostVector(agent_calls=2),
        ),
        (
            EventType.MODEL_CALLED,
            {"model_id": "model", "call_id": "model-0"},
            {"token_accounting": "measured"},
            CostVector(),
        ),
        (
            EventType.TOOL_CALLED,
            {"tool_name": "crop", "call_id": "tool-0"},
            {},
            CostVector(tool_calls=2),
        ),
    ],
)
def test_physical_call_event_must_own_exactly_one_counter(
    event_type, action, result, cost
) -> None:
    with pytest.raises(ValueError, match="must own exactly one"):
        EpisodeEvent(
            episode_id="episode-1",
            event_id="physical-call",
            seq=0,
            event_type=event_type,
            actor="runtime",
            action=action,
            result=result,
            cost_delta=cost,
        )


def test_invalid_action_counter_has_only_error_event_owner() -> None:
    with pytest.raises(ValueError, match="cannot own cost fields"):
        EpisodeEvent(
            episode_id="episode-1",
            event_id="warning-invalid",
            seq=0,
            event_type=EventType.WARNING,
            actor="runtime",
            action={},
            result={},
            cost_delta=CostVector(invalid_actions=1),
        )


def test_complete_trace_rejects_reused_physical_call_id() -> None:
    events = (
        EpisodeEvent(
            episode_id="episode-1",
            event_id="start",
            seq=0,
            event_type=EventType.EPISODE_STARTED,
            actor="runtime",
            action={"task_id": "task-1"},
            result={"observation_id": "observation-0"},
        ),
        EpisodeEvent(
            episode_id="episode-1",
            event_id="dispatch",
            seq=1,
            event_type=EventType.AGENT_DISPATCHED,
            actor="runtime",
            action={
                "agent_id": "generalist",
                "assignment_id": "assignment-0",
                "call_id": "physical-0",
            },
            result={},
            cost_delta=CostVector(agent_calls=1),
        ),
        EpisodeEvent(
            episode_id="episode-1",
            event_id="model",
            seq=2,
            event_type=EventType.MODEL_CALLED,
            actor="model",
            action={"model_id": "model", "call_id": "physical-0"},
            result={"token_accounting": "measured"},
            cost_delta=CostVector(model_calls=1),
        ),
        EpisodeEvent(
            episode_id="episode-1",
            event_id="finish",
            seq=3,
            event_type=EventType.EPISODE_FINISHED,
            actor="runtime",
            action={},
            result={"stop_reason": "aborted"},
        ),
    )
    with pytest.raises(ValueError, match="call_id.*reused"):
        EpisodeTrace.replay(events).validate_complete()


def test_complete_trace_replays_lifecycle_and_answer_reference() -> None:
    events = (
        EpisodeEvent(
            episode_id="episode-1",
            event_id="start",
            seq=0,
            event_type=EventType.EPISODE_STARTED,
            actor="runtime",
            action={"task_id": "task-1"},
            result={"observation_id": "observation-0"},
            elapsed_ms=0,
        ),
        EpisodeEvent(
            episode_id="episode-1",
            event_id="record",
            seq=1,
            event_type=EventType.RECORD_ADDED,
            actor="runtime",
            action={},
            result={
                "records": [
                    {"record_id": "answer-1", "record_type": "answer_candidate"}
                ]
            },
            elapsed_ms=2,
        ),
        EpisodeEvent(
            episode_id="episode-1",
            event_id="finish",
            seq=2,
            event_type=EventType.EPISODE_FINISHED,
            actor="runtime",
            action={"answer_candidate_ref": "answer-1"},
            result={"stop_reason": "completed"},
            elapsed_ms=3,
        ),
    )
    state = EpisodeTrace.replay(events).validate_complete()
    assert state.answer_candidate_ref == "answer-1"
    assert state.wall_time_ms == 3


def test_complete_trace_rejects_missing_start_and_dangling_answer() -> None:
    finish = EpisodeEvent(
        episode_id="episode-1",
        event_id="finish",
        seq=0,
        event_type=EventType.EPISODE_FINISHED,
        actor="runtime",
        action={"answer_candidate_ref": "missing"},
        result={"stop_reason": "completed"},
    )
    with pytest.raises(ValueError, match="first event"):
        EpisodeTrace("episode-1", (finish,)).validate_complete()


def test_event_rejects_unknown_schema_and_trace_rejects_nonmonotonic_time() -> None:
    payload = make_event(0).to_dict()
    payload["schema_version"] = "future.v9"
    with pytest.raises(ValueError, match="schema_version"):
        EpisodeEvent.from_dict(payload)

    first = make_event(0)
    second_payload = make_event(1).to_dict()
    second_payload["elapsed_ms"] = 0.0
    second = EpisodeEvent.from_dict(second_payload)
    late_first_payload = first.to_dict()
    late_first_payload["elapsed_ms"] = 2.0
    late_first = EpisodeEvent.from_dict(late_first_payload)
    with pytest.raises(ValueError, match="monotonic"):
        EpisodeTrace("episode-1", (late_first, second))
