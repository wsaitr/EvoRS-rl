from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True, slots=True)
class VisualInput:
    """Runtime-only visual input. ``image`` must never be written to JSON logs."""

    visual_id: str
    label: str
    image: Any
    source_bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.visual_id.strip() or not self.label.strip():
            raise ValueError("visual_id and label must be non-empty")
        if self.source_bbox is not None:
            if len(self.source_bbox) != 4:
                raise ValueError("source_bbox must contain four normalized coordinates")
            values = tuple(float(value) for value in self.source_bbox)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("source_bbox values must be finite")
            x1, y1, x2, y2 = values
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                raise ValueError("source_bbox must be a valid normalized box")
            object.__setattr__(self, "source_bbox", values)

    def descriptor(self) -> dict[str, Any]:
        return {
            "visual_id": self.visual_id,
            "label": self.label,
            "source_bbox": list(self.source_bbox) if self.source_bbox is not None else None,
        }


class TokenAccounting(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class BackendUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    stop_reason: str | None = None
    token_accounting: TokenAccounting = TokenAccounting.UNAVAILABLE

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        latency = float(self.latency_ms)
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("latency_ms must be a non-negative finite number")
        object.__setattr__(self, "latency_ms", latency)
        if not isinstance(self.token_accounting, TokenAccounting):
            object.__setattr__(self, "token_accounting", TokenAccounting(self.token_accounting))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
            "token_accounting": self.token_accounting.value,
        }


@dataclass(frozen=True, slots=True)
class BackendRequest:
    prompt: str
    visuals: tuple[VisualInput, ...]
    structured_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        object.__setattr__(self, "visuals", tuple(self.visuals))
        if not self.visuals:
            raise ValueError("at least one visual input is required")

@dataclass
class AgentOutput:
    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    usage: BackendUsage = field(default_factory=BackendUsage)
    model_id: str = ""
    model_revision: str = ""

class ModelBackend(Protocol):
    def generate(self, image: Any, prompt: str, structured_context: Optional[dict[str, Any]] = None) -> AgentOutput:
        ...

    def encode_roi(self, image: Any, bbox: tuple[float, float, float, float]) -> Any | None:
        ...

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        ...
