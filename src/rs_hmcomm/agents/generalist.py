"""Single Generalist Agent used by the Phase 1 baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs_hmcomm.agents.action_protocol import (
    AgentAction,
    AgentActionParseError,
    Finish,
    parse_agent_action,
)
from rs_hmcomm.backends.base import (
    AgentOutput,
    BackendRequest,
    ModelBackend,
    VisualInput,
)
from rs_hmcomm.contracts import Evidence, TaskContext


@dataclass(frozen=True)
class GeneralistTurnObservation:
    task: TaskContext
    step: int
    visuals: tuple[VisualInput, ...]
    evidence: tuple[Evidence, ...] = ()
    recoverable_errors: tuple[str, ...] = ()
    require_tool: bool = False

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        object.__setattr__(self, "visuals", tuple(self.visuals))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "recoverable_errors", tuple(self.recoverable_errors))
        if not self.visuals:
            raise ValueError("Generalist observation requires at least one visual")

    def structured_context(self) -> dict[str, Any]:
        return {
            "protocol": "evors.generalist.v1",
            "step": self.step,
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "answer_format": self.task.answer_format,
            "visuals": [visual.descriptor() for visual in self.visuals],
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_ids": [item.evidence_id for item in self.evidence],
            "recoverable_errors": list(self.recoverable_errors),
            "require_tool": self.require_tool,
            # Used only by the deterministic MockBackend fixture.
            "mock_answer": self.task.metadata.get("mock_answer", "6"),
        }


@dataclass(frozen=True)
class GeneralistDecision:
    action: AgentAction
    backend_output: AgentOutput


class GeneralistDecisionError(AgentActionParseError):
    """A protocol failure that retains usage while hiding free-form reasoning."""

    def __init__(self, message: str, backend_output: AgentOutput) -> None:
        self.backend_output = backend_output
        super().__init__(message)


class GeneralistAgent:
    """A single VLM-backed Agent; it cannot call another Agent."""

    agent_id = "generalist"

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    def build_prompt(self, observation: GeneralistTurnObservation) -> str:
        visual_lines = "\n".join(
            f"- {item.visual_id}: {item.label}, bbox={item.source_bbox}"
            for item in observation.visuals
        )
        evidence_lines = "\n".join(
            f"- {item.evidence_id}: {item.summary} (artifacts={list(item.artifact_refs)})"
            for item in observation.evidence
        ) or "- none"
        error_lines = "\n".join(f"- {item}" for item in observation.recoverable_errors) or "- none"
        tool_requirement = (
            "For this smoke run, call the crop tool at least once before finishing."
            if observation.require_tool and not observation.evidence
            else "Use crop only when an ROI would materially improve the answer."
        )
        return f"""You are the single EvoRS-Comm Generalist for a remote-sensing image task.
Question: {observation.task.question}
Answer format: {observation.task.answer_format}

Available visuals:
{visual_lines}

Visual-use rule:
- `input_image` is the one canonical task scene.
- Any `crop:*` visual is a derived view of that same scene, not an additional
  scene or additional objects. Never add or double-count objects across the
  original image and its crop.
- The image inputs are supplied in the same order as the visual list above.

Available evidence records:
{evidence_lines}

Recoverable protocol/tool errors from the previous turn:
{error_lines}

{tool_requirement}
Return exactly one JSON object and no markdown or hidden reasoning.

To inspect an ROI:
{{"type":"call_tool","call_id":"unique-id","tool_name":"crop","arguments":{{"image_ref":"visual-id","bbox":[x1,y1,x2,y2]}}}}
Coordinates are normalized and must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.

To answer:
{{"type":"finish","answer":"short answer","evidence_ids":["existing-evidence-id"],"confidence":0.0}}
Only cite evidence IDs listed above. Do not invent tool names, visual IDs, or evidence IDs.
"""

    def build_evolution_prompt(
        self,
        observation: GeneralistTurnObservation,
        *,
        parent_answer: str,
        parent_evidence_ids: tuple[str, ...],
        parent_confidence: float | None,
    ) -> str:
        """Build one private answer-variation request after a legal Finish.

        This request is still part of the current Generalist activation.  It
        may revise the answer, cited evidence, or confidence, but it cannot
        call tools or publish a record directly.
        """

        evidence_lines = "\n".join(
            f"- {item.evidence_id}: {item.summary}"
            for item in observation.evidence
        ) or "- none"
        return f"""You are privately revising one candidate inside the active EvoRS Generalist.
Question: {observation.task.question}
Answer format: {observation.task.answer_format}

Current candidate:
- answer: {parent_answer}
- evidence_ids: {list(parent_evidence_ids)}
- confidence: {parent_confidence}

Visible evidence:
{evidence_lines}

Reinspect the supplied visuals and propose exactly one useful variation. Keep
the current answer when the evidence does not justify a change. You may only
cite evidence IDs listed above. This is private candidate search: do not call
a tool and do not emit commentary or hidden reasoning.

Return exactly one JSON object:
{{"type":"finish","answer":"short answer","evidence_ids":["existing-evidence-id"],"confidence":0.0}}
"""

    def _generate_request(self, request: BackendRequest) -> AgentOutput:
        generate_turn = getattr(self.backend, "generate_turn", None)
        if callable(generate_turn):
            return generate_turn(request)
        latest = request.visuals[-1]
        return self.backend.generate(
            latest.image,
            request.prompt,
            structured_context=dict(request.structured_context),
        )

    def propose_answer_variation(
        self,
        observation: GeneralistTurnObservation,
        *,
        parent_answer: str,
        parent_evidence_ids: tuple[str, ...],
        parent_confidence: float | None,
    ) -> GeneralistDecision:
        """Generate one private Finish-shaped answer variation."""

        context = observation.structured_context()
        context.update(
            {
                "protocol": "evors.generalist.answer_evolution.v1",
                "parent_candidate": {
                    "answer": parent_answer,
                    "evidence_ids": list(parent_evidence_ids),
                    "confidence": parent_confidence,
                },
            }
        )
        request = BackendRequest(
            prompt=self.build_evolution_prompt(
                observation,
                parent_answer=parent_answer,
                parent_evidence_ids=parent_evidence_ids,
                parent_confidence=parent_confidence,
            ),
            visuals=observation.visuals,
            structured_context=context,
        )
        output = self._generate_request(request)
        try:
            action = parse_agent_action(output.text)
        except AgentActionParseError as exc:
            raise GeneralistDecisionError(str(exc), output) from exc
        if not isinstance(action, Finish):
            raise GeneralistDecisionError(
                "answer evolution must propose a finish action",
                output,
            )
        return GeneralistDecision(action=action, backend_output=output)

    def decide(self, observation: GeneralistTurnObservation) -> GeneralistDecision:
        request = BackendRequest(
            prompt=self.build_prompt(observation),
            visuals=observation.visuals,
            structured_context=observation.structured_context(),
        )
        output = self._generate_request(request)
        try:
            action = parse_agent_action(output.text)
        except AgentActionParseError as exc:
            raise GeneralistDecisionError(str(exc), output) from exc
        return GeneralistDecision(action=action, backend_output=output)


__all__ = [
    "GeneralistAgent",
    "GeneralistDecision",
    "GeneralistDecisionError",
    "GeneralistTurnObservation",
]
