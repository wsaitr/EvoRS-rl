"""Phase 0 Agent boundary contracts.

These types describe the hand-off around an Agent call.  They do not introduce
a shared Workspace yet; Phase 3 will reduce validated ``AgentResult`` records
into that state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ._validation import (
    coerce_enum,
    json_object_copy,
    normalize_id_tuple,
    normalize_json_object,
    require_id,
    require_step,
    strict_dict,
)
from .runtime import BudgetLimits, CostVector, StopReason
from .task import (
    AnswerCandidate,
    Assignment,
    CapabilityRequest,
    Claim,
    Evidence,
    TaskContext,
    Verdict,
)


def _unique_record_ids(records: tuple[Any, ...], id_attribute: str, field_name: str) -> None:
    record_ids = [getattr(record, id_attribute) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{field_name} contains duplicate record IDs")


def _require_global_id_uniqueness(pairs: tuple[tuple[str, str], ...]) -> None:
    owners: dict[str, str] = {}
    for record_id, owner in pairs:
        previous = owners.get(record_id)
        if previous is not None:
            raise ValueError(
                f"record ID {record_id!r} is used by both {previous} and {owner}; "
                "IDs share one global namespace"
            )
        owners[record_id] = owner


class AgentResultStatus(str, Enum):
    COMPLETED = "completed"
    NEEDS_CAPABILITY = "needs_capability"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class AgentObservation:
    observation_id: str
    agent_id: str
    task: TaskContext
    assignment: Assignment
    step: int
    budget_limits: BudgetLimits
    cost_so_far: CostVector = CostVector()
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    answer_candidates: tuple[AnswerCandidate, ...] = ()
    verdicts: tuple[Verdict, ...] = ()
    inbox_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.observation_id, "observation_id")
        require_id(self.agent_id, "agent_id")
        require_step(self.step, "step")
        if not isinstance(self.budget_limits, BudgetLimits):
            raise TypeError("budget_limits must be a BudgetLimits")
        if not isinstance(self.cost_so_far, CostVector):
            raise TypeError("cost_so_far must be a CostVector")
        # An observation is a runtime boundary, so an already-invalid budget
        # state must be rejected before it reaches an Agent.
        self.budget_limits.require_within(self.cost_so_far)
        if self.assignment.agent_id != self.agent_id:
            raise ValueError("assignment.agent_id must match observation agent_id")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "answer_candidates", tuple(self.answer_candidates))
        object.__setattr__(self, "verdicts", tuple(self.verdicts))
        object.__setattr__(self, "inbox_refs", normalize_id_tuple(self.inbox_refs, "inbox_refs"))
        _unique_record_ids(self.evidence, "evidence_id", "evidence")
        _unique_record_ids(self.claims, "claim_id", "claims")
        _unique_record_ids(self.answer_candidates, "candidate_id", "answer_candidates")
        _unique_record_ids(self.verdicts, "verdict_id", "verdicts")
        _require_global_id_uniqueness(
            (
                (self.task.task_id, "task"),
                (self.assignment.assignment_id, "assignment"),
                *((item.evidence_id, "evidence") for item in self.evidence),
                *((item.claim_id, "claim") for item in self.claims),
                *((item.candidate_id, "answer_candidate") for item in self.answer_candidates),
                *((item.verdict_id, "verdict") for item in self.verdicts),
            )
        )
        evidence_ids = {item.evidence_id for item in self.evidence}
        claim_ids = {item.claim_id for item in self.claims}
        candidate_ids = {item.candidate_id for item in self.answer_candidates}
        for claim in self.claims:
            missing = set(claim.evidence_ids) - evidence_ids
            if missing:
                raise ValueError(f"claim {claim.claim_id} has dangling evidence refs: {sorted(missing)}")
        for candidate in self.answer_candidates:
            missing = (set(candidate.evidence_ids) - evidence_ids) | (
                set(candidate.claim_ids) - claim_ids
            )
            if missing:
                raise ValueError(
                    f"answer candidate {candidate.candidate_id} has dangling refs: {sorted(missing)}"
                )
        for verdict in self.verdicts:
            if verdict.target_ref not in evidence_ids | claim_ids | candidate_ids:
                raise ValueError(f"verdict {verdict.verdict_id} has dangling target_ref")
            missing = set(verdict.evidence_ids) - evidence_ids
            if missing:
                raise ValueError(
                    f"verdict {verdict.verdict_id} has dangling evidence refs: {sorted(missing)}"
                )
        visible_ids = self.visible_record_ids
        missing_context = set(self.assignment.context_refs) - visible_ids
        missing_inbox = set(self.inbox_refs) - visible_ids
        if missing_context:
            raise ValueError(f"assignment has dangling context refs: {sorted(missing_context)}")
        if missing_inbox:
            raise ValueError(f"inbox has dangling refs: {sorted(missing_inbox)}")

    @property
    def visible_record_ids(self) -> frozenset[str]:
        return frozenset(
            {
                self.task.task_id,
                self.assignment.assignment_id,
                *(item.evidence_id for item in self.evidence),
                *(item.claim_id for item in self.claims),
                *(item.candidate_id for item in self.answer_candidates),
                *(item.verdict_id for item in self.verdicts),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "agent_id": self.agent_id,
            "task": self.task.to_dict(),
            "assignment": self.assignment.to_dict(),
            "step": self.step,
            "budget_limits": self.budget_limits.to_dict(),
            "cost_so_far": self.cost_so_far.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "answer_candidates": [item.to_dict() for item in self.answer_candidates],
            "verdicts": [item.to_dict() for item in self.verdicts],
            "inbox_refs": list(self.inbox_refs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentObservation":
        values = strict_dict(
            data,
            cls.__name__,
            required={
                "observation_id",
                "agent_id",
                "task",
                "assignment",
                "step",
                "budget_limits",
            },
            optional={
                "cost_so_far",
                "evidence",
                "claims",
                "answer_candidates",
                "verdicts",
                "inbox_refs",
            },
        )
        return cls(
            observation_id=values["observation_id"],
            agent_id=values["agent_id"],
            task=TaskContext.from_dict(values["task"]),
            assignment=Assignment.from_dict(values["assignment"]),
            step=values["step"],
            budget_limits=BudgetLimits.from_dict(values["budget_limits"]),
            cost_so_far=CostVector.from_dict(values.get("cost_so_far", {})),
            evidence=tuple(Evidence.from_dict(item) for item in values.get("evidence", [])),
            claims=tuple(Claim.from_dict(item) for item in values.get("claims", [])),
            answer_candidates=tuple(
                AnswerCandidate.from_dict(item)
                for item in values.get("answer_candidates", [])
            ),
            verdicts=tuple(Verdict.from_dict(item) for item in values.get("verdicts", [])),
            inbox_refs=values.get("inbox_refs", ()),
        )


@dataclass(frozen=True)
class AgentResult:
    result_id: str
    agent_id: str
    assignment_id: str
    status: AgentResultStatus
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    answer_candidates: tuple[AnswerCandidate, ...] = ()
    verdicts: tuple[Verdict, ...] = ()
    capability_requests: tuple[CapabilityRequest, ...] = ()
    stop_reason: StopReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.result_id, "result_id")
        require_id(self.agent_id, "agent_id")
        require_id(self.assignment_id, "assignment_id")
        object.__setattr__(self, "status", coerce_enum(self.status, AgentResultStatus, "status"))
        if self.stop_reason is not None:
            object.__setattr__(
                self,
                "stop_reason",
                coerce_enum(self.stop_reason, StopReason, "stop_reason"),
            )
        for field_name in (
            "evidence",
            "claims",
            "answer_candidates",
            "verdicts",
            "capability_requests",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        _unique_record_ids(self.evidence, "evidence_id", "evidence")
        _unique_record_ids(self.claims, "claim_id", "claims")
        _unique_record_ids(self.answer_candidates, "candidate_id", "answer_candidates")
        _unique_record_ids(self.verdicts, "verdict_id", "verdicts")
        _unique_record_ids(self.capability_requests, "request_id", "capability_requests")
        _require_global_id_uniqueness(
            (
                *((item.evidence_id, "evidence") for item in self.evidence),
                *((item.claim_id, "claim") for item in self.claims),
                *((item.candidate_id, "answer_candidate") for item in self.answer_candidates),
                *((item.verdict_id, "verdict") for item in self.verdicts),
                *((item.request_id, "capability_request") for item in self.capability_requests),
            )
        )
        object.__setattr__(self, "metadata", normalize_json_object(self.metadata, "metadata"))
        produced = (
            *self.evidence,
            *self.claims,
            *self.answer_candidates,
            *self.verdicts,
            *self.capability_requests,
        )
        if any(item.created_by != self.agent_id for item in produced):
            raise ValueError("every produced record must have created_by equal to agent_id")
        if self.status == AgentResultStatus.NEEDS_CAPABILITY and not self.capability_requests:
            raise ValueError("needs_capability results require a CapabilityRequest")
        if self.status in {AgentResultStatus.COMPLETED, AgentResultStatus.NEEDS_CAPABILITY}:
            if self.stop_reason is not None:
                raise ValueError(f"{self.status.value} results must not set stop_reason")
        elif self.stop_reason is None:
            raise ValueError(f"{self.status.value} results require stop_reason")
        if self.status == AgentResultStatus.NEEDS_CAPABILITY and self.answer_candidates:
            raise ValueError("needs_capability results cannot also submit a final answer candidate")

    @property
    def produced_record_ids(self) -> frozenset[str]:
        return frozenset(
            {
                *(item.evidence_id for item in self.evidence),
                *(item.claim_id for item in self.claims),
                *(item.candidate_id for item in self.answer_candidates),
                *(item.verdict_id for item in self.verdicts),
                *(item.request_id for item in self.capability_requests),
            }
        )

    def validate_against(
        self,
        observation: AgentObservation,
        *,
        observed_cost_delta: CostVector | None = None,
    ) -> None:
        """Validate semantic references against the exact scoped observation."""

        if self.agent_id != observation.agent_id:
            raise ValueError("result agent_id does not match observation")
        if self.assignment_id != observation.assignment.assignment_id:
            raise ValueError("result assignment_id does not match observation")
        if observed_cost_delta is not None:
            observation.budget_limits.require_within(
                observation.cost_so_far,
                observed_cost_delta,
            )
        for item in (
            *self.evidence,
            *self.claims,
            *self.answer_candidates,
            *self.verdicts,
            *self.capability_requests,
        ):
            if item.created_step != observation.step:
                raise ValueError(
                    f"record created_step must equal observation step {observation.step}"
                )
        overlap = observation.visible_record_ids & self.produced_record_ids
        if overlap:
            raise ValueError(f"result attempts to replace visible records: {sorted(overlap)}")

        evidence_ids = {
            *(item.evidence_id for item in observation.evidence),
            *(item.evidence_id for item in self.evidence),
        }
        claim_ids = {
            *(item.claim_id for item in observation.claims),
            *(item.claim_id for item in self.claims),
        }
        candidate_ids = {
            *(item.candidate_id for item in observation.answer_candidates),
            *(item.candidate_id for item in self.answer_candidates),
        }
        allowed_context_ids = {
            *observation.visible_record_ids,
            *(item.evidence_id for item in self.evidence),
            *(item.claim_id for item in self.claims),
            *(item.candidate_id for item in self.answer_candidates),
            *(item.verdict_id for item in self.verdicts),
        }

        for claim in self.claims:
            missing = set(claim.evidence_ids) - evidence_ids
            if missing:
                raise ValueError(f"claim {claim.claim_id} has dangling evidence refs: {sorted(missing)}")
        for candidate in self.answer_candidates:
            missing_evidence = set(candidate.evidence_ids) - evidence_ids
            missing_claims = set(candidate.claim_ids) - claim_ids
            if missing_evidence or missing_claims:
                raise ValueError(
                    f"answer candidate {candidate.candidate_id} has dangling refs: "
                    f"{sorted(missing_evidence | missing_claims)}"
                )
        for verdict in self.verdicts:
            missing_evidence = set(verdict.evidence_ids) - evidence_ids
            if verdict.target_ref not in evidence_ids | claim_ids | candidate_ids:
                raise ValueError(f"verdict {verdict.verdict_id} has dangling target_ref")
            if missing_evidence:
                raise ValueError(
                    f"verdict {verdict.verdict_id} has dangling evidence refs: "
                    f"{sorted(missing_evidence)}"
                )
        for request in self.capability_requests:
            missing = set(request.context_refs) - allowed_context_ids
            if missing:
                raise ValueError(
                    f"capability request {request.request_id} has dangling context refs: "
                    f"{sorted(missing)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "agent_id": self.agent_id,
            "assignment_id": self.assignment_id,
            "status": self.status.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "answer_candidates": [item.to_dict() for item in self.answer_candidates],
            "verdicts": [item.to_dict() for item in self.verdicts],
            "capability_requests": [item.to_dict() for item in self.capability_requests],
            "stop_reason": self.stop_reason.value if self.stop_reason is not None else None,
            "metadata": json_object_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentResult":
        values = strict_dict(
            data,
            cls.__name__,
            required={"result_id", "agent_id", "assignment_id", "status"},
            optional={
                "evidence",
                "claims",
                "answer_candidates",
                "verdicts",
                "capability_requests",
                "stop_reason",
                "metadata",
            },
        )
        return cls(
            result_id=values["result_id"],
            agent_id=values["agent_id"],
            assignment_id=values["assignment_id"],
            status=values["status"],
            evidence=tuple(Evidence.from_dict(item) for item in values.get("evidence", [])),
            claims=tuple(Claim.from_dict(item) for item in values.get("claims", [])),
            answer_candidates=tuple(
                AnswerCandidate.from_dict(item)
                for item in values.get("answer_candidates", [])
            ),
            verdicts=tuple(Verdict.from_dict(item) for item in values.get("verdicts", [])),
            capability_requests=tuple(
                CapabilityRequest.from_dict(item)
                for item in values.get("capability_requests", [])
            ),
            stop_reason=values.get("stop_reason"),
            metadata=values.get("metadata", {}),
        )


__all__ = ["AgentObservation", "AgentResult", "AgentResultStatus"]
