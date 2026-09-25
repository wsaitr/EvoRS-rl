"""Pure control-layer contracts for non-RL and learned routing policies.

The controller receives descriptors only. Raw Workspace records, image
objects, evidence payloads, bounding boxes, and latent values remain outside
the control boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from rs_hmcomm.contracts import (
    BudgetLimits,
    CapabilityRequest,
    CostVector,
    RecordStatus,
    StopReason,
    TaskContext,
)
from rs_hmcomm.contracts._validation import (
    normalize_id_tuple,
    require_id,
    require_step,
    require_text,
)
from rs_hmcomm.workspace import WorkspaceState


_MODALITIES = frozenset({"text", "structure", "latent"})
_SPATIAL_LEVELS = frozenset({"scene", "region", "group", "object"})


@dataclass(frozen=True, slots=True)
class ControllerRecordSummary:
    """A routable record descriptor with no task-bearing content payload."""

    record_id: str
    record_type: str
    created_by: str
    status: str | None = None
    subtype: str | None = None
    spatial_level: str | None = None
    confidence: float | None = None
    byte_size: int = 0
    compatible_consumers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.record_id, "record_id")
        require_text(self.record_type, "record_type")
        require_id(self.created_by, "created_by")
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")
        object.__setattr__(
            self,
            "compatible_consumers",
            normalize_id_tuple(self.compatible_consumers, "compatible_consumers"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "created_by": self.created_by,
            "status": self.status,
            "subtype": self.subtype,
            "spatial_level": self.spatial_level,
            "confidence": self.confidence,
            "byte_size": self.byte_size,
            "compatible_consumers": list(self.compatible_consumers),
        }


@dataclass(frozen=True, slots=True)
class ControllerWorkspaceSummary:
    """Compact Workspace index exposed to the Evolution Controller."""

    episode_id: str
    version: int
    records: tuple[ControllerRecordSummary, ...] = ()
    message_count: int = 0
    total_cost: CostVector = field(default_factory=CostVector)

    def __post_init__(self) -> None:
        require_id(self.episode_id, "episode_id")
        if isinstance(self.version, bool) or self.version < 0:
            raise ValueError("version must be non-negative")
        normalized = tuple(self.records)
        if any(not isinstance(item, ControllerRecordSummary) for item in normalized):
            raise TypeError("records must contain ControllerRecordSummary values")
        if len({item.record_id for item in normalized}) != len(normalized):
            raise ValueError("controller summary record IDs must be unique")
        object.__setattr__(self, "records", normalized)
        if isinstance(self.message_count, bool) or self.message_count < 0:
            raise ValueError("message_count must be non-negative")
        if not isinstance(self.total_cost, CostVector):
            raise TypeError("total_cost must be a CostVector")

    @classmethod
    def from_workspace(
        cls,
        workspace: WorkspaceState,
        *,
        message_count: int = 0,
        total_cost: CostVector | None = None,
    ) -> "ControllerWorkspaceSummary":
        if not isinstance(workspace, WorkspaceState):
            raise TypeError("workspace must be a WorkspaceState")
        records: list[ControllerRecordSummary] = []
        records.extend(
            ControllerRecordSummary(
                record_id=item.evidence_id,
                record_type="evidence",
                created_by=item.created_by,
                subtype=item.kind,
                confidence=item.confidence,
            )
            for item in workspace.evidence
        )
        records.extend(
            ControllerRecordSummary(
                record_id=item.claim_id,
                record_type="claim",
                created_by=item.created_by,
                status=item.status.value,
                confidence=item.confidence,
            )
            for item in workspace.claims
        )
        records.extend(
            ControllerRecordSummary(
                record_id=item.candidate_id,
                record_type="answer_candidate",
                created_by=item.created_by,
                status=item.status.value,
                confidence=item.confidence,
            )
            for item in workspace.answer_candidates
        )
        records.extend(
            ControllerRecordSummary(
                record_id=item.node_id,
                record_type="spatial_node",
                created_by=str(item.provenance.get("created_by", "workspace")),
                spatial_level=item.level.value,
                confidence=item.confidence,
            )
            for item in workspace.spatial_nodes
        )
        records.extend(
            ControllerRecordSummary(
                record_id=item.artifact_id,
                record_type="artifact",
                created_by="artifact-store",
                subtype=item.kind.value,
                byte_size=item.byte_size,
                compatible_consumers=item.compatible_consumers,
            )
            for item in workspace.artifacts
        )
        return cls(
            episode_id=workspace.episode_id,
            version=workspace.version,
            records=tuple(records),
            message_count=message_count,
            total_cost=total_cost or CostVector(),
        )

    @property
    def record_ids(self) -> frozenset[str]:
        return frozenset(item.record_id for item in self.records)

    def records_of_type(self, *record_types: str) -> tuple[ControllerRecordSummary, ...]:
        requested = frozenset(record_types)
        return tuple(item for item in self.records if item.record_type in requested)

    @property
    def verified_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            item.record_id
            for item in self.records
            if item.record_type == "answer_candidate"
            and item.status == RecordStatus.VERIFIED.value
        )

    def to_dict(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for item in self.records:
            counts[item.record_type] = counts.get(item.record_type, 0) + 1
        return {
            "episode_id": self.episode_id,
            "version": self.version,
            "record_counts": counts,
            "records": [item.to_dict() for item in self.records],
            "message_count": self.message_count,
            "total_cost": self.total_cost.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContextSpec:
    """Exact Workspace disclosure requested for one Expert activation."""

    modalities: tuple[str, ...] = ("text", "structure")
    payload_refs: tuple[str, ...] = ()
    focus_node_ids: tuple[str, ...] = ()
    spatial_level: str | None = None
    max_items: int = 64

    def __post_init__(self) -> None:
        if isinstance(self.modalities, (str, bytes)):
            raise TypeError("modalities must be a tuple of modality names")
        modalities = tuple(str(item) for item in self.modalities)
        if not modalities or any(item not in _MODALITIES for item in modalities):
            raise ValueError("modalities may contain only text, structure, and latent")
        if len(set(modalities)) != len(modalities):
            raise ValueError("modalities must not contain duplicates")
        order = {"text": 0, "structure": 1, "latent": 2}
        object.__setattr__(self, "modalities", tuple(sorted(modalities, key=order.__getitem__)))
        object.__setattr__(self, "payload_refs", normalize_id_tuple(self.payload_refs, "payload_refs"))
        object.__setattr__(self, "focus_node_ids", normalize_id_tuple(self.focus_node_ids, "focus_node_ids"))
        if self.spatial_level is not None and self.spatial_level not in _SPATIAL_LEVELS:
            raise ValueError("spatial_level must be scene, region, group, object, or None")
        if isinstance(self.max_items, bool) or not isinstance(self.max_items, int) or self.max_items < 1:
            raise ValueError("max_items must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "modalities": list(self.modalities),
            "payload_refs": list(self.payload_refs),
            "focus_node_ids": list(self.focus_node_ids),
            "spatial_level": self.spatial_level,
            "max_items": self.max_items,
        }


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    expert_id: str
    objective: str
    context_spec: ContextSpec = field(default_factory=ContextSpec)
    local_budget: BudgetLimits | None = None
    policy_action_id: str | None = None

    def __post_init__(self) -> None:
        require_id(self.expert_id, "expert_id")
        require_text(self.objective, "objective")
        if not isinstance(self.context_spec, ContextSpec):
            raise TypeError("context_spec must be a ContextSpec")
        if self.local_budget is not None and not isinstance(self.local_budget, BudgetLimits):
            raise TypeError("local_budget must be BudgetLimits or None")
        if self.policy_action_id is not None:
            require_id(self.policy_action_id, "policy_action_id")


@dataclass(frozen=True, slots=True)
class FinishDecision:
    answer_candidate_ref: str
    policy_action_id: str | None = None

    def __post_init__(self) -> None:
        require_id(self.answer_candidate_ref, "answer_candidate_ref")
        if self.policy_action_id is not None:
            require_id(self.policy_action_id, "policy_action_id")


@dataclass(frozen=True, slots=True)
class AbortDecision:
    stop_reason: StopReason
    policy_action_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stop_reason, StopReason):
            try:
                object.__setattr__(self, "stop_reason", StopReason(self.stop_reason))
            except (TypeError, ValueError) as exc:
                raise ValueError("stop_reason must be a StopReason") from exc
        if self.policy_action_id is not None:
            require_id(self.policy_action_id, "policy_action_id")


ControllerDecision: TypeAlias = DispatchDecision | FinishDecision | AbortDecision


@dataclass(frozen=True, slots=True)
class ControllerObservation:
    task: TaskContext
    workspace: ControllerWorkspaceSummary | WorkspaceState
    agent_path: tuple[str, ...]
    step: int
    max_steps: int
    capability_requests: tuple[CapabilityRequest, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskContext):
            raise TypeError("task must be a TaskContext")
        summary = self.workspace
        if isinstance(summary, WorkspaceState):
            if summary.task.task_id != self.task.task_id:
                raise ValueError("workspace task must match the controller task")
            summary = ControllerWorkspaceSummary.from_workspace(summary)
            object.__setattr__(self, "workspace", summary)
        elif not isinstance(summary, ControllerWorkspaceSummary):
            raise TypeError("workspace must be a ControllerWorkspaceSummary")
        if isinstance(self.agent_path, str) or not isinstance(self.agent_path, (tuple, list)):
            raise TypeError("agent_path must be a tuple or list of expert IDs")
        path = tuple(require_id(item, f"agent_path[{index}]") for index, item in enumerate(self.agent_path))
        object.__setattr__(self, "agent_path", path)
        require_step(self.step, "step")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        requests = tuple(self.capability_requests)
        if any(not isinstance(item, CapabilityRequest) for item in requests):
            raise TypeError("capability_requests must contain CapabilityRequest values")
        object.__setattr__(self, "capability_requests", requests)


class RoutingPolicy(Protocol):
    """Select one control decision from the current compact state view."""

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        ...


__all__ = [
    "AbortDecision",
    "ContextSpec",
    "ControllerDecision",
    "ControllerObservation",
    "ControllerRecordSummary",
    "ControllerWorkspaceSummary",
    "DispatchDecision",
    "FinishDecision",
    "RoutingPolicy",
]
