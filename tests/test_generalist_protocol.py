from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest
from PIL import Image

from rs_hmcomm.agents.action_protocol import (
    AgentActionParseError,
    CallTool,
    Finish,
    parse_agent_action,
)
from rs_hmcomm.agents.generalist import GeneralistAgent, GeneralistTurnObservation
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.backends.mock import MockBackend
from rs_hmcomm.contracts import TaskContext


def test_generalist_prompt_marks_crops_as_derived_views() -> None:
    prompt = GeneralistAgent(MockBackend()).build_prompt(
        GeneralistTurnObservation(
            task=TaskContext(
                task_id="task-1",
                image_ref="fixture://airport",
                question="How many aircraft?",
                task_type="vqa",
                answer_format="short_text",
            ),
            step=1,
            visuals=(
                VisualInput("input_image", "input_image", Image.new("RGB", (8, 8))),
                VisualInput(
                    "visual_crop-1",
                    "crop:crop-1",
                    Image.new("RGB", (4, 4)),
                    (0.0, 0.0, 0.5, 0.5),
                ),
            ),
        )
    )

    assert "one canonical task scene" in prompt
    assert "Never add or double-count objects" in prompt


def test_parse_call_tool_action() -> None:
    action = parse_agent_action(
        json.dumps(
            {
                "type": "call_tool",
                "call_id": "call_1",
                "tool_name": "crop",
                "arguments": {
                    "image_ref": "input_image",
                    "bbox": [0.2, 0.1, 0.8, 0.9],
                },
            }
        )
    )

    assert action == CallTool(
        call_id="call_1",
        tool_name="crop",
        arguments={
            "image_ref": "input_image",
            "bbox": [0.2, 0.1, 0.8, 0.9],
        },
    )


def test_parse_finish_action_normalizes_ids_and_confidence() -> None:
    action = parse_agent_action(
        '{"type":"finish","answer":"six","evidence_ids":["ev_1"],"confidence":1}'
    )

    assert action == Finish("six", ("ev_1",), 1.0)
    assert isinstance(action.evidence_ids, tuple)


def test_action_records_are_frozen() -> None:
    action = Finish("six", ("ev_1",), 0.9)
    with pytest.raises(FrozenInstanceError):
        action.answer = "seven"  # type: ignore[misc]


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{\"type\":\"finish\"}\n```",
        "{'type': 'finish'}",
        '{"type":"finish"}{"type":"finish"}',
        '[{"type":"finish"}]',
        '{"type":"finish","answer":"x","evidence_ids":["ev"],"confidence":NaN}',
        '{"type":"finish","answer":"x","evidence_ids":["ev"],"confidence":Infinity}',
        '{"type":"finish","type":"call_tool"}',
    ],
)
def test_parser_rejects_non_pure_or_non_standard_json(raw: str) -> None:
    with pytest.raises(AgentActionParseError):
        parse_agent_action(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"finish","answer":"six","evidence_ids":["ev"],"extra":1}',
        '{"type":"call_tool","call_id":"c","tool_name":"crop","arguments":{},"extra":1}',
        '{"type":"unknown","answer":"six","evidence_ids":["ev"]}',
    ],
)
def test_parser_rejects_unknown_fields_and_action_types(raw: str) -> None:
    with pytest.raises(AgentActionParseError):
        parse_agent_action(raw)


@pytest.mark.parametrize(
    "evidence_ids",
    [[], [""], ["   "], ["ev_1", "ev_1"]],
)
def test_finish_rejects_empty_or_duplicate_evidence_refs(evidence_ids: list[str]) -> None:
    raw = json.dumps(
        {
            "type": "finish",
            "answer": "six",
            "evidence_ids": evidence_ids,
        }
    )
    with pytest.raises(AgentActionParseError):
        parse_agent_action(raw)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, True, "0.5"])
def test_finish_rejects_invalid_confidence(confidence: object) -> None:
    raw = json.dumps(
        {
            "type": "finish",
            "answer": "six",
            "evidence_ids": ["ev_1"],
            "confidence": confidence,
        }
    )
    with pytest.raises(AgentActionParseError):
        parse_agent_action(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"call_tool","call_id":"","tool_name":"crop","arguments":{}}',
        '{"type":"call_tool","call_id":"c","tool_name":"","arguments":{}}',
        '{"type":"call_tool","call_id":"c","tool_name":"crop","arguments":[]}',
        '{"type":"finish","answer":" ","evidence_ids":["ev_1"]}',
    ],
)
def test_parser_rejects_invalid_field_types_or_empty_text(raw: str) -> None:
    with pytest.raises(AgentActionParseError):
        parse_agent_action(raw)
