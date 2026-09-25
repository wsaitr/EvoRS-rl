"""Serializable domain records shared across EvoRS-Comm phases.

These records carry only lightweight values and opaque references.  Images,
crops, tensors and latent representations are deliberately excluded and must
be resolved by the runtime from ``image_ref`` or ``artifact_refs``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ._validation import (
    coerce_enum,
    json_object_copy,
    normalize_confidence,
    normalize_id_tuple,
    normalize_json_object,
    normalize_text_tuple,
    require_id,
    require_step,
    require_text,
    strict_dict,
)


class RecordStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class VerdictDecision(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    NEEDS_EVIDENCE = "needs_evidence"


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    image_ref: str
    question: str
    task_type: str = "vqa"
    answer_format: str = "free_text"
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.task_id, "task_id")
        require_text(self.image_ref, "image_ref")
        require_text(self.question, "question")
        require_text(self.task_type, "task_type")
        require_text(self.answer_format, "answer_format")
        choices = normalize_text_tuple(self.choices, "choices")
        if self.answer_format in {"choice", "multiple_choice"} and len(choices) < 2:
            raise ValueError("choice tasks require at least two choices")
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "metadata", normalize_json_object(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "image_ref": self.image_ref,
            "question": self.question,
            "task_type": self.task_type,
            "answer_format": self.answer_format,
            "choices": list(self.choices),
            "metadata": json_object_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskContext":
        values = strict_dict(
            data,
            cls.__name__,
            required={"task_id", "image_ref", "question"},
            optional={"task_type", "answer_format", "choices", "metadata"},
        )
        return cls(
            task_id=values["task_id"],
            image_ref=values["image_ref"],
            question=values["question"],
            task_type=values.get("task_type", "vqa"),
            answer_format=values.get("answer_format", "free_text"),
            choices=values.get("choices", ()),
            metadata=values.get("metadata", {}),
        )


@dataclass(frozen=True)
class Assignment:
    assignment_id: str
    agent_id: str
    objective: str
    created_step: int = 0
    focus_node_ids: tuple[str, ...] = ()
    context_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.assignment_id, "assignment_id")
        require_id(self.agent_id, "agent_id")
        require_text(self.objective, "objective")
        require_step(self.created_step)
        object.__setattr__(self, "focus_node_ids", normalize_id_tuple(self.focus_node_ids, "focus_node_ids"))
        object.__setattr__(self, "context_refs", normalize_id_tuple(self.context_refs, "context_refs"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "agent_id": self.agent_id,
            "objective": self.objective,
            "created_step": self.created_step,
            "focus_node_ids": list(self.focus_node_ids),
            "context_refs": list(self.context_refs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Assignment":
        values = strict_dict(
            data,
            cls.__name__,
            required={"assignment_id", "agent_id", "objective"},
            optional={"created_step", "focus_node_ids", "context_refs"},
        )
        return cls(
            assignment_id=values["assignment_id"],
            agent_id=values["agent_id"],
            objective=values["objective"],
            created_step=values.get("created_step", 0),
            focus_node_ids=values.get("focus_node_ids", ()),
            context_refs=values.get("context_refs", ()),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: str
    summary: str
    created_by: str
    created_step: int = 0
    confidence: float | None = None
    spatial_node_ids: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.evidence_id, "evidence_id")
        require_text(self.kind, "kind")
        require_text(self.summary, "summary")
        require_id(self.created_by, "created_by")
        require_step(self.created_step)
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(self, "spatial_node_ids", normalize_id_tuple(self.spatial_node_ids, "spatial_node_ids"))
        object.__setattr__(self, "artifact_refs", normalize_id_tuple(self.artifact_refs, "artifact_refs"))
        object.__setattr__(self, "payload", normalize_json_object(self.payload, "payload"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "summary": self.summary,
            "created_by": self.created_by,
            "created_step": self.created_step,
            "confidence": self.confidence,
            "spatial_node_ids": list(self.spatial_node_ids),
            "artifact_refs": list(self.artifact_refs),
            "payload": json_object_copy(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        values = strict_dict(
            data,
            cls.__name__,
            required={"evidence_id", "kind", "summary", "created_by"},
            optional={"created_step", "confidence", "spatial_node_ids", "artifact_refs", "payload"},
        )
        return cls(
            evidence_id=values["evidence_id"],
            kind=values["kind"],
            summary=values["summary"],
            created_by=values["created_by"],
            created_step=values.get("created_step", 0),
            confidence=values.get("confidence"),
            spatial_node_ids=values.get("spatial_node_ids", ()),
            artifact_refs=values.get("artifact_refs", ()),
            payload=values.get("payload", {}),
        )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    statement: str
    created_by: str
    evidence_ids: tuple[str, ...]
    created_step: int = 0
    confidence: float | None = None
    status: RecordStatus = RecordStatus.PROPOSED

    def __post_init__(self) -> None:
        require_id(self.claim_id, "claim_id")
        require_text(self.statement, "statement")
        require_id(self.created_by, "created_by")
        require_step(self.created_step)
        object.__setattr__(self, "evidence_ids", normalize_id_tuple(self.evidence_ids, "evidence_ids", require_nonempty=True))
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(self, "status", coerce_enum(self.status, RecordStatus, "status"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "created_by": self.created_by,
            "evidence_ids": list(self.evidence_ids),
            "created_step": self.created_step,
            "confidence": self.confidence,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Claim":
        values = strict_dict(
            data,
            cls.__name__,
            required={"claim_id", "statement", "created_by", "evidence_ids"},
            optional={"created_step", "confidence", "status"},
        )
        return cls(
            claim_id=values["claim_id"],
            statement=values["statement"],
            created_by=values["created_by"],
            evidence_ids=values["evidence_ids"],
            created_step=values.get("created_step", 0),
            confidence=values.get("confidence"),
            status=values.get("status", RecordStatus.PROPOSED),
        )


@dataclass(frozen=True)
class AnswerCandidate:
    candidate_id: str
    answer: str
    created_by: str
    claim_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    created_step: int = 0
    confidence: float | None = None
    status: RecordStatus = RecordStatus.PROPOSED

    def __post_init__(self) -> None:
        require_id(self.candidate_id, "candidate_id")
        require_text(self.answer, "answer")
        require_id(self.created_by, "created_by")
        require_step(self.created_step)
        claim_ids = normalize_id_tuple(self.claim_ids, "claim_ids")
        evidence_ids = normalize_id_tuple(self.evidence_ids, "evidence_ids")
        if not claim_ids and not evidence_ids:
            raise ValueError("AnswerCandidate must reference at least one claim or evidence record")
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(self, "status", coerce_enum(self.status, RecordStatus, "status"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "answer": self.answer,
            "created_by": self.created_by,
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "created_step": self.created_step,
            "confidence": self.confidence,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnswerCandidate":
        values = strict_dict(
            data,
            cls.__name__,
            required={"candidate_id", "answer", "created_by"},
            optional={"claim_ids", "evidence_ids", "created_step", "confidence", "status"},
        )
        return cls(
            candidate_id=values["candidate_id"],
            answer=values["answer"],
            created_by=values["created_by"],
            claim_ids=values.get("claim_ids", ()),
            evidence_ids=values.get("evidence_ids", ()),
            created_step=values.get("created_step", 0),
            confidence=values.get("confidence"),
            status=values.get("status", RecordStatus.PROPOSED),
        )


@dataclass(frozen=True)
class Verdict:
    verdict_id: str
    target_ref: str
    decision: VerdictDecision
    reason: str
    created_by: str
    evidence_ids: tuple[str, ...] = ()
    created_step: int = 0
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_id(self.verdict_id, "verdict_id")
        require_id(self.target_ref, "target_ref")
        object.__setattr__(self, "decision", coerce_enum(self.decision, VerdictDecision, "decision"))
        require_text(self.reason, "reason")
        require_id(self.created_by, "created_by")
        require_step(self.created_step)
        object.__setattr__(self, "evidence_ids", normalize_id_tuple(self.evidence_ids, "evidence_ids"))
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_id": self.verdict_id,
            "target_ref": self.target_ref,
            "decision": self.decision.value,
            "reason": self.reason,
            "created_by": self.created_by,
            "evidence_ids": list(self.evidence_ids),
            "created_step": self.created_step,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Verdict":
        values = strict_dict(
            data,
            cls.__name__,
            required={"verdict_id", "target_ref", "decision", "reason", "created_by"},
            optional={"evidence_ids", "created_step", "confidence"},
        )
        return cls(
            verdict_id=values["verdict_id"],
            target_ref=values["target_ref"],
            decision=values["decision"],
            reason=values["reason"],
            created_by=values["created_by"],
            evidence_ids=values.get("evidence_ids", ()),
            created_step=values.get("created_step", 0),
            confidence=values.get("confidence"),
        )


@dataclass(frozen=True)
class CapabilityRequest:
    request_id: str
    capability: str
    objective: str
    created_by: str
    created_step: int = 0
    focus_node_ids: tuple[str, ...] = ()
    context_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.request_id, "request_id")
        require_text(self.capability, "capability")
        require_text(self.objective, "objective")
        require_id(self.created_by, "created_by")
        require_step(self.created_step)
        object.__setattr__(self, "focus_node_ids", normalize_id_tuple(self.focus_node_ids, "focus_node_ids"))
        object.__setattr__(self, "context_refs", normalize_id_tuple(self.context_refs, "context_refs"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "capability": self.capability,
            "objective": self.objective,
            "created_by": self.created_by,
            "created_step": self.created_step,
            "focus_node_ids": list(self.focus_node_ids),
            "context_refs": list(self.context_refs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityRequest":
        values = strict_dict(
            data,
            cls.__name__,
            required={"request_id", "capability", "objective", "created_by"},
            optional={"created_step", "focus_node_ids", "context_refs"},
        )
        return cls(
            request_id=values["request_id"],
            capability=values["capability"],
            objective=values["objective"],
            created_by=values["created_by"],
            created_step=values.get("created_step", 0),
            focus_node_ids=values.get("focus_node_ids", ()),
            context_refs=values.get("context_refs", ()),
        )


__all__ = [
    "AnswerCandidate",
    "Assignment",
    "CapabilityRequest",
    "Claim",
    "Evidence",
    "RecordStatus",
    "TaskContext",
    "Verdict",
    "VerdictDecision",
]
