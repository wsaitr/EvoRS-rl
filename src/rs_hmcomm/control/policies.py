"""Auditable non-RL routing policies for Phase 4."""
from __future__ import annotations

import re

from rs_hmcomm.contracts import RecordStatus, StopReason

from .contracts import (
    AbortDecision,
    ContextSpec,
    ControllerDecision,
    ControllerObservation,
    ControllerWorkspaceSummary,
    DispatchDecision,
    FinishDecision,
)


_OBJECTIVES = {
    "visual": "Gather question-relevant visual evidence with private active perception",
    "knowledge": "Provide task-relevant remote-sensing domain guidance",
    "reasoning": "Reason over scoped evidence and propose an answer candidate",
    "verification": "Verify the proposed answer against image and disclosed evidence",
}

_COUNT_ENGLISH = re.compile(r"\bcount(?:ing|s|ed)?\b|\bhow\s+many\b")
_CLASSIFY_ENGLISH = re.compile(r"\bclassification\b|\bclassify\b|\btype\b")


def _records(
    summary: ControllerWorkspaceSummary,
    *,
    types: tuple[str, ...] = (),
    creators: tuple[str, ...] = (),
    statuses: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return tuple(
        item.record_id
        for item in summary.records
        if (not types or item.record_type in types)
        and (not creators or item.created_by in creators)
        and (not statuses or item.status in statuses)
    )


def _context_for(
    expert_id: str,
    summary: ControllerWorkspaceSummary,
) -> ContextSpec:
    if expert_id == "visual":
        return ContextSpec(modalities=("text", "structure"), max_items=16)

    if expert_id == "knowledge":
        refs = _records(summary, types=("evidence", "spatial_node"), creators=("visual", "workspace"))
        focus = _records(summary, types=("spatial_node",))
        return ContextSpec(
            modalities=("text", "structure"),
            payload_refs=refs[:24],
            focus_node_ids=focus[:12],
            max_items=36,
        )

    if expert_id == "reasoning":
        refs = _records(summary, types=("evidence", "spatial_node"))
        artifacts = tuple(
            item.record_id
            for item in summary.records
            if item.record_type == "artifact"
            and expert_id in item.compatible_consumers
        )
        selected = tuple(dict.fromkeys((*refs, *artifacts)))[:64]
        modalities = ("text", "structure", "latent") if artifacts else ("text", "structure")
        focus = _records(summary, types=("spatial_node",))[:16]
        return ContextSpec(
            modalities=modalities,
            payload_refs=selected,
            focus_node_ids=focus,
            max_items=64,
        )

    proposed = _records(
        summary,
        types=("answer_candidate",),
        statuses=(RecordStatus.PROPOSED.value,),
    )
    # Granting the candidate lets ObservationBuilder close over only its
    # supporting claims/evidence/artifacts.
    refs = proposed[-1:] if proposed else ()
    return ContextSpec(
        modalities=("text", "structure"),
        payload_refs=refs,
        max_items=16,
    )


def _dispatch(expert_id: str, observation: ControllerObservation) -> DispatchDecision:
    return DispatchDecision(
        expert_id,
        _OBJECTIVES[expert_id],
        context_spec=_context_for(expert_id, observation.workspace),
    )


def _finish_or_abort(observation: ControllerObservation) -> ControllerDecision:
    verified = observation.workspace.verified_candidate_ids
    if not verified:
        return AbortDecision(StopReason.NO_PROGRESS)
    return FinishDecision(verified[-1])


def _follow_sequence(
    observation: ControllerObservation,
    sequence: tuple[str, ...],
) -> ControllerDecision:
    if observation.step >= observation.max_steps:
        return AbortDecision(StopReason.MAX_STEPS)
    next_index = 0
    for executed in observation.agent_path:
        if next_index < len(sequence) and executed == sequence[next_index]:
            next_index += 1
    if next_index < len(sequence):
        return _dispatch(sequence[next_index], observation)
    return _finish_or_abort(observation)


class RuleRoutingPolicy:
    """Task- and Workspace-conditioned dynamic routing baseline."""

    counting_sequence = ("visual", "reasoning", "verification")
    classification_sequence = ("visual", "knowledge", "reasoning", "verification")

    @classmethod
    def _sequence_for(cls, question: str) -> tuple[str, ...]:
        normalized = question.casefold()
        if any(marker in normalized for marker in ("统计", "多少", "几")) or _COUNT_ENGLISH.search(normalized):
            return cls.counting_sequence
        if "机场类型" in normalized or _CLASSIFY_ENGLISH.search(normalized):
            return cls.classification_sequence
        return cls.counting_sequence

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        if observation.step >= observation.max_steps:
            return AbortDecision(StopReason.MAX_STEPS)
        summary = observation.workspace
        if summary.verified_candidate_ids:
            return FinishDecision(summary.verified_candidate_ids[-1])
        # Compatibility for policy-only sequence probes that advance an
        # artificial agent_path without applying expert deltas. Real runtime
        # episodes always advance Workspace version with each successful step.
        if summary.version == 0 and observation.agent_path:
            return _follow_sequence(
                observation,
                self._sequence_for(observation.task.question),
            )

        visual_evidence = _records(summary, types=("evidence",), creators=("visual",))
        if not visual_evidence:
            return _dispatch("visual", observation)

        sequence = self._sequence_for(observation.task.question)
        knowledge_evidence = _records(summary, types=("evidence",), creators=("knowledge",))
        if "knowledge" in sequence and not knowledge_evidence:
            return _dispatch("knowledge", observation)

        proposed = _records(
            summary,
            types=("answer_candidate",),
            statuses=(RecordStatus.PROPOSED.value,),
        )
        if not proposed:
            return _dispatch("reasoning", observation)
        return _dispatch("verification", observation)


class FixedWorkflowPolicy:
    """Always run the fixed core visual/reasoning/verification workflow."""

    sequence = ("visual", "reasoning", "verification")

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        return _follow_sequence(observation, self.sequence)


class AllAgentPolicy:
    """Always activate every cognitive expert in one fixed order."""

    sequence = ("visual", "knowledge", "reasoning", "verification")

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        return _follow_sequence(observation, self.sequence)


__all__ = [
    "AllAgentPolicy",
    "FixedWorkflowPolicy",
    "RuleRoutingPolicy",
    "_context_for",
]
