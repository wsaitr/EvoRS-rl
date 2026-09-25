from __future__ import annotations

import pytest

from rs_hmcomm.contracts.runtime import (
    BudgetLimits,
    CostVector,
    EpisodeEvent,
    EpisodeTrace,
    EventType,
)
from rs_hmcomm.observability import EventJsonlLogger, load_event_traces, read_events


def event(episode_id: str, seq: int) -> EpisodeEvent:
    return EpisodeEvent(
        episode_id=episode_id,
        event_id=f"{episode_id}-{seq}",
        seq=seq,
        event_type=EventType.TOOL_CALLED,
        actor="crop",
        action={"tool_name": "crop", "call_id": f"call-{seq}", "bbox": [0.0, 0.0, 1.0, 1.0]},
        result={"evidence_ref": f"evidence-{seq}"},
        cost_delta=CostVector(tool_calls=1, billed_compute_ms=3.0),
        elapsed_ms=float(seq),
    )


def test_jsonl_logger_reads_and_replays_interleaved_episodes(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventJsonlLogger(path, append=False)
    logger.write_event(event("episode-a", 0))
    logger.write_event(event("episode-b", 0))
    logger.write_event(event("episode-a", 1))

    assert logger.count == 3
    assert len(read_events(path)) == 3
    traces = load_event_traces(path)
    assert [trace.episode_id for trace in traces] == ["episode-a", "episode-b"]
    assert traces[0].total_cost.tool_calls == 2
    assert traces[1].total_cost.tool_calls == 1


def test_jsonl_event_id_is_scoped_by_episode(tmp_path) -> None:
    path = tmp_path / "scoped-events.jsonl"
    logger = EventJsonlLogger(path, append=False)
    first = event("episode-a", 0)
    second_payload = event("episode-b", 0).to_dict()
    second_payload["event_id"] = first.event_id
    second = EpisodeEvent.from_dict(second_payload)

    logger.write_event(first)
    logger.write_event(second)
    assert len(load_event_traces(path)) == 2

    duplicate_payload = event("episode-a", 1).to_dict()
    duplicate_payload["event_id"] = first.event_id
    duplicate = EpisodeEvent.from_dict(duplicate_payload)
    with pytest.raises(ValueError, match="duplicate event key"):
        logger.write_event(duplicate)


def test_jsonl_logger_can_write_a_trace(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = EpisodeTrace("episode-a").append(event("episode-a", 0))
    EventJsonlLogger(path, append=False).write_trace(trace)
    assert load_event_traces(path) == (trace,)


def test_jsonl_roundtrip_restores_declared_trace_budget(tmp_path) -> None:
    path = tmp_path / "budgeted-trace.jsonl"
    budget = BudgetLimits(max_agent_calls=1, max_model_calls=2)
    events = (
        EpisodeEvent(
            episode_id="episode-budget",
            event_id="start",
            seq=0,
            event_type=EventType.EPISODE_STARTED,
            actor="runtime",
            action={"task_id": "task-budget"},
            result={
                "observation_id": "observation-budget",
                "budget_limits": budget.to_dict(),
            },
        ),
        EpisodeEvent(
            episode_id="episode-budget",
            event_id="finish",
            seq=1,
            event_type=EventType.EPISODE_FINISHED,
            actor="runtime",
            action={},
            result={"stop_reason": "aborted"},
        ),
    )
    trace = EpisodeTrace.replay(events, budget_limits=budget)

    EventJsonlLogger(path, append=False).write_trace(trace)
    restored = load_event_traces(path)[0]

    assert restored.budget_limits == budget
    assert restored == trace
    assert restored.validate_complete().budget_violations == ()


def test_reader_reports_corrupt_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1|line 2"):
        read_events(path)
