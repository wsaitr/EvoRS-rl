from __future__ import annotations
from typing import Any, Optional
from .base import AgentOutput

class MockBackend:
    def generate(self, image: Any, prompt: str, structured_context: Optional[dict] = None) -> AgentOutput:
        p = prompt.lower()
        if "global" in p:
            return AgentOutput(
                "Possible airport-like region in center.",
                {"semantic": "airport_like", "bbox": [0.20, 0.20, 0.85, 0.85], "level": "region"},
                0.72,
            )
        if "local" in p:
            return AgentOutput(
                "Detected an aircraft group in the target region.",
                {"semantic": "aircraft_group", "bbox": [0.45, 0.40, 0.70, 0.68], "level": "group"},
                0.81,
            )
        if "hierarchy" in p:
            return AgentOutput(
                "Aircraft group is inside apron; region supports airport interpretation.",
                {"relation": ["aircraft_group", "inside", "apron"]},
                0.83,
            )
        if "verify" in p or "verifier" in p:
            return AgentOutput("Evidence is sufficient for a tentative answer.", {"verified": True}, 0.88)
        return AgentOutput("Residual analysis: no additional evidence.", {}, 0.5)

    def encode_roi(self, image: Any, bbox: tuple[float, float, float, float]) -> None:
        return None
