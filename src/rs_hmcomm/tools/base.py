"""Runtime boundary shared by Phase 1 tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.contracts import Evidence
from rs_hmcomm.contracts._validation import (
    json_object_copy,
    normalize_json_object,
    require_id,
    require_step,
    require_text,
)


class ToolError(RuntimeError):
    """Base class for safe, expected tool failures."""


class ToolInputError(ToolError, ValueError):
    """Raised when a tool call has invalid or unavailable input."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    agent_id: str
    step: int
    visuals_by_id: Mapping[str, VisualInput]

    def __post_init__(self) -> None:
        require_id(self.agent_id, "agent_id")
        require_step(self.step, "step")
        if not isinstance(self.visuals_by_id, Mapping):
            raise TypeError("visuals_by_id must be a mapping")
        copied: dict[str, VisualInput] = {}
        for visual_id, visual in self.visuals_by_id.items():
            require_id(visual_id, "visuals_by_id key")
            if not isinstance(visual, VisualInput):
                raise TypeError("visuals_by_id values must be VisualInput instances")
            if visual.visual_id != visual_id:
                raise ValueError("visuals_by_id keys must match VisualInput.visual_id")
            copied[visual_id] = visual
        object.__setattr__(self, "visuals_by_id", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool outcome with runtime media separated from JSON-safe log data."""

    call_id: str
    tool_name: str
    ok: bool
    observation: str
    evidence: tuple[Evidence, ...] = ()
    visuals: tuple[VisualInput, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        require_id(self.call_id, "call_id")
        require_id(self.tool_name, "tool_name")
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be a bool")
        require_text(self.observation, "observation")

        evidence = tuple(self.evidence)
        visuals = tuple(self.visuals)
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence must contain Evidence records")
        if any(not isinstance(item, VisualInput) for item in visuals):
            raise TypeError("visuals must contain VisualInput runtime objects")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("evidence contains duplicate evidence_id values")
        if len({item.visual_id for item in visuals}) != len(visuals):
            raise ValueError("visuals contains duplicate visual_id values")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "visuals", visuals)
        object.__setattr__(
            self,
            "metadata",
            normalize_json_object(self.metadata, "metadata"),
        )

        if self.ok and self.error is not None:
            raise ValueError("successful ToolResult must not contain an error")
        if not self.ok:
            if self.error is None:
                raise ValueError("failed ToolResult requires an error")
            require_text(self.error, "error")

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view that never includes PIL images or tensors."""

        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "observation": self.observation,
            "evidence": [item.to_dict() for item in self.evidence],
            "visuals": [item.descriptor() for item in self.visuals],
            "metadata": json_object_copy(self.metadata),
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for the serializable/logging view; runtime visuals stay excluded."""

        return self.to_log_dict()


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def execute(self, call: CallTool, context: ToolContext) -> ToolResult:
        ...


__all__ = [
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolInputError",
    "ToolResult",
]
