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
from rs_hmcomm.backends.mock import MockBackend
from rs_hmcomm.contracts import BudgetLimits, EventType, StopReason
from rs_hmcomm.runtime import SingleAgentRuntime
from rs_hmcomm.tools import CropTool, ToolRegistry


def generous_budget(**overrides) -> BudgetLimits:
    values = {
        "max_agent_calls": 1,
        "max_tool_calls": 2,
        "max_model_calls": 4,
        "max_model_input_tokens": 10000,
        "max_model_output_tokens": 10000,
        "max_billed_compute_ms": 30000,
        "max_wall_time_ms": 30000,
        "max_errors": 2,
        "max_retries": 1,
        "max_invalid_actions": 2,
    }
    values.update(overrides)
    return BudgetLimits(**values)


def runtime(backend, *, budget: BudgetLimits | None = None, max_steps: int = 4):
    return SingleAgentRuntime(
        GeneralistAgent(backend),
        ToolRegistry((CropTool(),)),
        budget or generous_budget(),
        max_steps=max_steps,
    )


def test_mock_generalist_runs_image_tool_evidence_answer_loop() -> None:
    budget = generous_budget()
    result = runtime(MockBackend(), budget=budget).run(
        Image.new("RGB", (100, 100), color="gray"),
        "How many aircraft are visible?",
        sample_id="mock-airport-1",
        require_tool=True,
        task_metadata={"mock_answer": "6", "fixture": True},
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert result.answer == "6"
    assert result.cost.agent_calls == 1
    assert result.cost.model_calls == 2
    assert result.cost.tool_calls == 1
    assert result.trace.budget_limits == budget
    assert result.trace.events[0].result["budget_limits"] == budget.to_dict()
    assert len(result.agent_result.evidence) == 1
    assert result.agent_result.answer_candidates[0].evidence_ids == (
        result.agent_result.evidence[0].evidence_id,
    )
    replay = result.trace.validate_complete()
    assert replay.answer_candidate_ref == "answer-generalist-0"
    assert replay.agent_path == ("generalist",)

    serialized = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "PIL" not in serialized
    assert "Image" not in serialized


def test_tool_budget_stops_before_crop_executes() -> None:
    result = runtime(MockBackend(), budget=generous_budget(max_tool_calls=0)).run(
        Image.new("RGB", (32, 32)),
        "Inspect the image",
        sample_id="tool-budget",
        require_tool=True,
    )

    assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert result.answer is None
    assert result.cost.model_calls == 1
    assert result.cost.tool_calls == 0
    assert not result.agent_result.evidence


class MalformedBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        return AgentOutput(
            text="not-json",
            usage=BackendUsage(
                input_tokens=5,
                output_tokens=1,
                latency_ms=0.1,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="malformed-spy",
            model_revision="v1",
        )


def test_malformed_model_action_is_costed_and_terminates_without_eval() -> None:
    result = runtime(
        MalformedBackend(),
        budget=generous_budget(max_retries=0),
    ).run(
        Image.new("RGB", (32, 32)),
        "Question?",
        sample_id="malformed",
    )

    assert result.stop_reason == StopReason.INVALID_ACTION
    assert result.answer is None
    assert result.cost.model_calls == 1
    assert result.cost.invalid_actions == 1
    assert result.cost.errors == 1
    model_event = next(
        event for event in result.trace.events if event.event_type == EventType.MODEL_CALLED
    )
    assert model_event.result["token_accounting"] == "measured"
    assert "not-json" not in json.dumps(result.trace.to_dict())


class RaisingBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        raise RuntimeError("backend unavailable")


def test_backend_exception_is_recorded_as_failed_result() -> None:
    result = runtime(RaisingBackend()).run(
        Image.new("RGB", (32, 32)),
        "Question?",
        sample_id="backend-error",
    )

    assert result.stop_reason == StopReason.BACKEND_ERROR
    assert result.agent_result.status.value == "failed"
    assert result.cost.model_calls == 1
    assert result.cost.errors == 1
    result.trace.validate_complete()


class DanglingFinishBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        return AgentOutput(
            text=json.dumps(
                {
                    "type": "finish",
                    "answer": "6",
                    "evidence_ids": ["invented-evidence"],
                }
            ),
            usage=BackendUsage(token_accounting=TokenAccounting.UNAVAILABLE),
            model_id="dangling-spy",
        )


def test_finish_cannot_cite_nonexistent_evidence() -> None:
    result = runtime(
        DanglingFinishBackend(),
        budget=generous_budget(max_retries=0),
    ).run(
        Image.new("RGB", (16, 16)),
        "Question?",
        sample_id="dangling-finish",
    )
    assert result.stop_reason == StopReason.INVALID_ACTION
    assert result.answer is None
    assert result.cost.invalid_actions == 1


class TokenOverrunBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        return AgentOutput(
            text=json.dumps(
                {
                    "type": "finish",
                    "answer": "unused",
                    "evidence_ids": ["not-reached"],
                }
            ),
            usage=BackendUsage(
                input_tokens=11,
                output_tokens=1,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="token-overrun-spy",
        )


def test_measured_token_overrun_finishes_budget_exhausted_and_replays_violation() -> None:
    budget = generous_budget(max_model_input_tokens=10)
    result = runtime(TokenOverrunBackend(), budget=budget).run(
        Image.new("RGB", (16, 16)),
        "Question?",
        sample_id="measured-token-overrun",
    )

    assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert result.answer is None
    assert result.trace.budget_limits == budget
    assert result.cost.model_input_tokens == 11
    replay = result.trace.validate_complete()
    assert replay.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert replay.budget_violations == ("model_input_tokens",)


class SequenceBackend:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = tuple(json.dumps(item) for item in responses)
        self.calls = 0

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        response = self.responses[self.calls]
        self.calls += 1
        return AgentOutput(
            text=response,
            usage=BackendUsage(
                input_tokens=1,
                output_tokens=1,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="sequence-spy",
        )


class CountingCropTool(CropTool):
    def __init__(self) -> None:
        self.execute_count = 0

    def execute(self, call, context):
        self.execute_count += 1
        return super().execute(call, context)


def test_duplicate_model_tool_call_id_is_not_executed_twice_and_retries() -> None:
    duplicate_call = {
        "type": "call_tool",
        "call_id": "crop-repeat",
        "tool_name": "crop",
        "arguments": {
            "image_ref": "input_image",
            "bbox": [0.1, 0.1, 0.9, 0.9],
        },
    }
    backend = SequenceBackend(
        [
            duplicate_call,
            duplicate_call,
            {
                "type": "finish",
                "answer": "6",
                "evidence_ids": ["evidence_crop-repeat"],
            },
        ]
    )
    tool = CountingCropTool()
    runner = SingleAgentRuntime(
        GeneralistAgent(backend),
        ToolRegistry((tool,)),
        generous_budget(),
        max_steps=4,
    )

    result = runner.run(
        Image.new("RGB", (32, 32)),
        "How many aircraft?",
        sample_id="duplicate-tool-call",
        require_tool=True,
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert backend.calls == 3
    assert tool.execute_count == 1
    assert result.cost.tool_calls == 1
    assert result.cost.invalid_actions == 1
    assert result.cost.retries == 1
    assert sum(
        event.event_type == EventType.TOOL_CALLED for event in result.trace.events
    ) == 1
    error = next(
        event for event in result.trace.events if event.event_type == EventType.ERROR
    )
    retry = next(
        event for event in result.trace.events if event.event_type == EventType.WARNING
    )
    assert error.action["reason"] == StopReason.INVALID_ACTION.value
    assert "not unique" in error.result["message"]
    assert retry.action["reason"] == "retry"
    result.validate_phase1_completion(require_tool=True)


class SecretRaisingBackend:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        raise RuntimeError(f"backend failed with {self.secret}")


def test_backend_exception_secret_is_not_serialized() -> None:
    secret = "BACKEND_SECRET_80e48a73"
    result = runtime(SecretRaisingBackend(secret)).run(
        Image.new("RGB", (16, 16)),
        "Question?",
        sample_id="secret-backend",
    )

    assert result.stop_reason == StopReason.BACKEND_ERROR
    assert secret not in json.dumps(result.trace.to_dict(), ensure_ascii=False)
    assert secret not in json.dumps(result.to_dict(), ensure_ascii=False)


class SingleToolCallBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        return AgentOutput(
            text=json.dumps(
                {
                    "type": "call_tool",
                    "call_id": "secret-tool-call",
                    "tool_name": "exploding",
                    "arguments": {},
                }
            ),
            usage=BackendUsage(token_accounting=TokenAccounting.MEASURED),
            model_id="tool-call-spy",
        )


class SecretExplodingTool:
    name = "exploding"
    description = "Raise a private test exception."

    def __init__(self, secret: str) -> None:
        self.secret = secret

    def execute(self, call, context):
        raise RuntimeError(f"tool failed with {self.secret}")


def test_tool_exception_secret_is_not_serialized() -> None:
    secret = "TOOL_SECRET_45cad190"
    runner = SingleAgentRuntime(
        GeneralistAgent(SingleToolCallBackend()),
        ToolRegistry((SecretExplodingTool(secret),)),
        generous_budget(),
    )
    result = runner.run(
        Image.new("RGB", (16, 16)),
        "Question?",
        sample_id="secret-tool",
    )

    assert result.stop_reason == StopReason.TOOL_ERROR
    assert secret not in json.dumps(result.trace.to_dict(), ensure_ascii=False)
    assert secret not in json.dumps(result.to_dict(), ensure_ascii=False)
