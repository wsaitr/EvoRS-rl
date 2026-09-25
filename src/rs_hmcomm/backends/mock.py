from __future__ import annotations
import json
from typing import Any, Optional
from .base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
)

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

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        """Deterministic Phase 1 protocol fixture, not a visual model."""

        context = dict(request.structured_context)
        if context.get("protocol") == "evors.generalist.answer_evolution.v1":
            parent = dict(context.get("parent_candidate", {}))
            parent_confidence = parent.get("confidence")
            confidence = (
                min(1.0, float(parent_confidence) + 0.05)
                if isinstance(parent_confidence, (int, float))
                and not isinstance(parent_confidence, bool)
                else 0.95
            )
            action = {
                "type": "finish",
                "answer": str(context.get("mock_answer", parent.get("answer", "6"))),
                "evidence_ids": list(parent.get("evidence_ids", ())),
                "confidence": confidence,
            }
            text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            return AgentOutput(
                text=text,
                usage=BackendUsage(
                    input_tokens=len(request.prompt.split()),
                    output_tokens=len(text.split()),
                    latency_ms=0.0,
                    stop_reason="mock_evolution_complete",
                    token_accounting=TokenAccounting.ESTIMATED,
                ),
                model_id="mock-generalist",
                model_revision="deterministic-v1",
            )
        if context.get("protocol") != "evors.generalist.v1":
            visual = request.visuals[-1]
            return self.generate(visual.image, request.prompt, context)

        step = context.get("step", 0)
        evidence_ids = tuple(context.get("evidence_ids", ()))
        if step == 0 or not evidence_ids:
            action = {
                "type": "call_tool",
                "call_id": f"crop-{step}",
                "tool_name": "crop",
                "arguments": {
                    "image_ref": request.visuals[-1].visual_id,
                    "bbox": [0.2, 0.2, 0.8, 0.8],
                },
            }
        else:
            action = {
                "type": "finish",
                "answer": str(context.get("mock_answer", "6")),
                "evidence_ids": list(evidence_ids),
                "confidence": 0.9,
            }
        text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        return AgentOutput(
            text=text,
            usage=BackendUsage(
                input_tokens=len(request.prompt.split()),
                output_tokens=len(text.split()),
                latency_ms=0.0,
                stop_reason="mock_complete",
                token_accounting=TokenAccounting.ESTIMATED,
            ),
            model_id="mock-generalist",
            model_revision="deterministic-v1",
        )
