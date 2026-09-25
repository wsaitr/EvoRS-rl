"""Strict JSON action protocol for the Phase 1 generalist agent.

The model is allowed to emit exactly one JSON object.  Markdown fences,
Python literals, duplicate keys, non-finite numbers and schema extensions are
rejected deliberately: tool execution must never depend on permissive parsing
or ``eval``-style behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, TypeAlias

from rs_hmcomm.contracts._validation import (
    normalize_confidence,
    normalize_id_tuple,
    normalize_json_object,
    require_id,
    require_text,
)


class AgentActionParseError(ValueError):
    """Raised when a model response is not a valid Phase 1 action."""


@dataclass(frozen=True, slots=True)
class CallTool:
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.call_id, "call_id")
        require_id(self.tool_name, "tool_name")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a JSON object")
        object.__setattr__(
            self,
            "arguments",
            normalize_json_object(self.arguments, "arguments"),
        )


@dataclass(frozen=True, slots=True)
class Finish:
    answer: str
    evidence_ids: tuple[str, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_text(self.answer, "answer")
        object.__setattr__(
            self,
            "evidence_ids",
            normalize_id_tuple(
                self.evidence_ids,
                "evidence_ids",
                require_nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            normalize_confidence(self.confidence),
        )


AgentAction: TypeAlias = CallTool | Finish


def _reject_non_standard_constant(value: str) -> None:
    raise AgentActionParseError(f"non-standard JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AgentActionParseError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_fields(
    data: Mapping[str, Any],
    *,
    action_type: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    keys = frozenset(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise AgentActionParseError(
            f"{action_type} action is missing required fields: {sorted(missing)}"
        )
    if unknown:
        raise AgentActionParseError(
            f"{action_type} action contains unknown fields: {sorted(unknown)}"
        )


def parse_agent_action(raw: str) -> AgentAction:
    """Parse one exact JSON object into a validated agent action.

    The discriminator is ``type`` and the only accepted values are
    ``call_tool`` and ``finish``.
    """

    if not isinstance(raw, str):
        raise TypeError("agent action must be provided as a JSON string")
    if not raw.strip():
        raise AgentActionParseError("agent action must not be empty")

    try:
        data = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_standard_constant,
        )
    except AgentActionParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AgentActionParseError("agent action must be one pure JSON object") from exc

    if not isinstance(data, dict):
        raise AgentActionParseError("agent action JSON must contain an object")
    action_type = data.get("type")
    if not isinstance(action_type, str):
        raise AgentActionParseError("agent action requires a string 'type' field")

    try:
        if action_type == "call_tool":
            _strict_fields(
                data,
                action_type=action_type,
                required=frozenset({"type", "call_id", "tool_name", "arguments"}),
            )
            return CallTool(
                call_id=data["call_id"],
                tool_name=data["tool_name"],
                arguments=data["arguments"],
            )

        if action_type == "finish":
            _strict_fields(
                data,
                action_type=action_type,
                required=frozenset({"type", "answer", "evidence_ids"}),
                optional=frozenset({"confidence"}),
            )
            return Finish(
                answer=data["answer"],
                evidence_ids=data["evidence_ids"],
                confidence=data.get("confidence"),
            )
    except AgentActionParseError:
        raise
    except (TypeError, ValueError) as exc:
        raise AgentActionParseError(f"invalid {action_type} action: {exc}") from exc

    raise AgentActionParseError(
        "agent action type must be one of ['call_tool', 'finish']"
    )


__all__ = [
    "AgentAction",
    "AgentActionParseError",
    "CallTool",
    "Finish",
    "parse_agent_action",
]
