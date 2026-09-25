"""Stable, JSON-serializable contracts for EvoRS-Comm."""

from .agent import AgentObservation, AgentResult, AgentResultStatus
from .experiment import DatasetSpec, canonical_manifest_digest, deterministic_sample_ids
from .metrics import (
    EvaluationTarget,
    MetricSummary,
    TaskScore,
    aggregate_scores,
    exact_match_score,
    normalize_short_answer,
)
from .runtime import (
    BudgetExceeded,
    BudgetLimits,
    CostEntry,
    CostLedger,
    CostVector,
    EpisodeEvent,
    EpisodeTrace,
    EventType,
    ReplayState,
    StopReason,
)
from .task import (
    AnswerCandidate,
    Assignment,
    CapabilityRequest,
    Claim,
    Evidence,
    RecordStatus,
    TaskContext,
    Verdict,
    VerdictDecision,
)

__all__ = [
    "AgentObservation",
    "AgentResult",
    "AgentResultStatus",
    "AnswerCandidate",
    "Assignment",
    "BudgetExceeded",
    "BudgetLimits",
    "CapabilityRequest",
    "Claim",
    "CostEntry",
    "CostLedger",
    "CostVector",
    "DatasetSpec",
    "Evidence",
    "EpisodeEvent",
    "EpisodeTrace",
    "EvaluationTarget",
    "EventType",
    "MetricSummary",
    "RecordStatus",
    "ReplayState",
    "StopReason",
    "TaskContext",
    "TaskScore",
    "Verdict",
    "VerdictDecision",
    "aggregate_scores",
    "canonical_manifest_digest",
    "deterministic_sample_ids",
    "exact_match_score",
    "normalize_short_answer",
]
