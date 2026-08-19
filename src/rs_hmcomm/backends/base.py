from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

@dataclass
class AgentOutput:
    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

class ModelBackend(Protocol):
    def generate(self, image: Any, prompt: str, structured_context: Optional[dict[str, Any]] = None) -> AgentOutput:
        ...

    def encode_roi(self, image: Any, bbox: tuple[float, float, float, float]) -> Any | None:
        ...
