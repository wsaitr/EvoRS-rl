from __future__ import annotations

import json

from PIL import Image

from rs_hmcomm.agents.generalist import GeneralistAgent
from rs_hmcomm.backends.base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
)
from rs_hmcomm.contracts import BudgetLimits, EpisodeTrace, EventType, StopReason
from rs_hmcomm.evolution import EvolutionBudget, EvolutionConfig
from rs_hmcomm.runtime import SingleAgentRuntime
from rs_hmcomm.tools import CropTool, ToolRegistry


def budget(*, max_model_calls: int = 4) -> BudgetLimits:
    return BudgetLimits(
        max_agent_calls=1,
        max_tool_calls=1,
        max_model_calls=max_model_calls,
        max_model_input_tokens=1000,
        max_model_output_tokens=1000,
        max_billed_compute_ms=10000,
        max_wall_time_ms=10000,
        max_errors=1,
        max_retries=0,
        max_invalid_actions=1,
    )


def one_plus_one(*, enabled: bool = True) -> EvolutionConfig:
    return EvolutionConfig(
        enabled=enabled,
        population_size=2 if enabled else 1,
        max_generations=1,
        no_progress_generations=1,
        seed=7,
        budget=EvolutionBudget(
            max_candidate_evaluations=2 if enabled else 1,
            max_variations=1 if enabled else 0,
        ),
    )


class EvolutionSequenceBackend:
    def __init__(self, *, include_variation: bool = True) -> None:
        self.include_variation = include_variation
        self.calls = 0
        self.protocols: list[str] = []

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        self.calls += 1
        protocol = str(request.structured_context.get("protocol"))
        self.protocols.append(protocol)
        if self.calls == 1:
            action = {
                "type": "call_tool",
                "call_id": "crop-1",
                "tool_name": "crop",
                "arguments": {
                    "image_ref": "input_image",
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                },
            }
        elif self.calls == 2:
            action = {
                "type": "finish",
                "answer": "seed answer",
                "evidence_ids": ["evidence_crop-1"],
                "confidence": 0.2,
            }
        elif self.calls == 3 and self.include_variation:
            action = {
                "type": "finish",
                "answer": "improved answer",
                "evidence_ids": ["evidence_crop-1"],
                "confidence": 0.9,
            }
        else:  # pragma: no cover - protects the fixture contract
            raise AssertionError("unexpected backend call")
        text = json.dumps(action, separators=(",", ":"))
        return AgentOutput(
            text=text,
            usage=BackendUsage(
                input_tokens=3,
                output_tokens=2,
                latency_ms=1.0,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="evolution-sequence-spy",
            model_revision="v1",
        )


def runner(
    backend: EvolutionSequenceBackend,
    *,
    evolution_config: EvolutionConfig | None,
    max_model_calls: int = 4,
) -> SingleAgentRuntime:
    return SingleAgentRuntime(
        GeneralistAgent(backend),
        ToolRegistry((CropTool(),)),
        budget(max_model_calls=max_model_calls),
        max_steps=3,
        evolution_config=evolution_config,
    )


def test_first_legal_finish_runs_private_one_plus_one_and_commits_only_winner() -> None:
    backend = EvolutionSequenceBackend()
    result = runner(backend, evolution_config=one_plus_one()).run(
        Image.new("RGB", (32, 32)),
        "What is visible?",
        sample_id="evolving-generalist",
        require_tool=True,
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert result.answer == "improved answer"
    assert backend.calls == 3
    assert backend.protocols == [
        "evors.generalist.v1",
        "evors.generalist.v1",
        "evors.generalist.answer_evolution.v1",
    ]
    assert result.cost.agent_calls == 1
    assert result.cost.tool_calls == 1
    assert result.cost.model_calls == 3

    event_types = [event.event_type for event in result.trace.events]
    assert event_types.count(EventType.CANDIDATE_PROPOSED) == 2
    assert event_types.count(EventType.CANDIDATE_EVALUATED) == 2
    assert event_types.count(EventType.CANDIDATE_SELECTED) == 1

    proposed = [
        event.action["candidate_id"]
        for event in result.trace.events
        if event.event_type == EventType.CANDIDATE_PROPOSED
    ]
    selected = next(
        event
        for event in result.trace.events
        if event.event_type == EventType.CANDIDATE_SELECTED
    )
    assert selected.result["selected_candidate_id"] == proposed[1]

    indexed_events = list(enumerate(result.trace.events))
    proposed_positions = [
        index
        for index, event in indexed_events
        if event.event_type == EventType.CANDIDATE_PROPOSED
    ]
    evaluated_positions = [
        index
        for index, event in indexed_events
        if event.event_type == EventType.CANDIDATE_EVALUATED
    ]
    model_positions = [
        index
        for index, event in indexed_events
        if event.event_type == EventType.MODEL_CALLED
    ]
    selected_position = next(
        index
        for index, event in indexed_events
        if event.event_type == EventType.CANDIDATE_SELECTED
    )
    finish_position = next(
        index
        for index, event in indexed_events
        if event.event_type == EventType.AGENT_ACTED
        and event.action.get("action_type") == "finish"
    )
    assert (
        proposed_positions[0]
        < evaluated_positions[0]
        < model_positions[-1]
        < proposed_positions[1]
        < evaluated_positions[1]
        < selected_position
        < finish_position
    )

    public_answer_records = [
        record
        for event in result.trace.events
        if event.event_type == EventType.RECORD_ADDED
        for record in event.result["records"]
        if record["record_type"] == "answer_candidate"
    ]
    assert [record["record_id"] for record in public_answer_records] == [
        "answer-generalist-0"
    ]
    assert not set(proposed) & {
        record["record_id"]
        for event in result.trace.events
        if event.event_type == EventType.RECORD_ADDED
        for record in event.result["records"]
    }
    assert result.agent_result.answer_candidates[0].answer == "improved answer"
    summary = result.agent_result.metadata["answer_evolution"]
    assert summary["metrics"]["extra_cost"]["model_calls"] == 1
    result.validate_phase1_completion(require_tool=True)
    EpisodeTrace.from_dict(result.trace.to_dict()).validate_complete()


def test_disabled_evolution_preserves_phase1_model_path_and_events() -> None:
    backend = EvolutionSequenceBackend(include_variation=False)
    result = runner(backend, evolution_config=one_plus_one(enabled=False)).run(
        Image.new("RGB", (32, 32)),
        "What is visible?",
        sample_id="disabled-evolution",
        require_tool=True,
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert result.answer == "seed answer"
    assert backend.calls == 2
    assert result.cost.model_calls == 2
    assert "answer_evolution" not in result.agent_result.metadata
    assert not any(
        event.event_type
        in {
            EventType.CANDIDATE_PROPOSED,
            EventType.CANDIDATE_EVALUATED,
            EventType.CANDIDATE_SELECTED,
        }
        for event in result.trace.events
    )


def test_evolution_model_call_is_preflighted_against_episode_budget() -> None:
    backend = EvolutionSequenceBackend(include_variation=False)
    result = runner(
        backend,
        evolution_config=one_plus_one(),
        max_model_calls=2,
    ).run(
        Image.new("RGB", (32, 32)),
        "What is visible?",
        sample_id="evolution-budget",
        require_tool=True,
    )

    assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert result.answer is None
    assert backend.calls == 2
    assert result.cost.model_calls == 2
    assert not result.agent_result.answer_candidates
    result.trace.validate_complete()
