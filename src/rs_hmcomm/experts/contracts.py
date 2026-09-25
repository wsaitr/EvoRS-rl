"""Stable Cognitive Expert boundary for Phase 4."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from rs_hmcomm.contracts import BudgetLimits, CapabilityRequest, CostVector
from rs_hmcomm.contracts._validation import (
    normalize_json_object,
    require_id,
    require_step,
    require_text,
)
from rs_hmcomm.workspace import ScopedWorkspaceView, WorkspaceDelta


EXPERT_IDS = frozenset({"visual", "reasoning", "verification", "knowledge"})


def _require_expert_id(value: str) -> str:
    require_id(value, "expert_id")
    if value not in EXPERT_IDS:
        raise ValueError(f"expert_id must be one of {sorted(EXPERT_IDS)}")
    return value


@dataclass(frozen=True, slots=True)
class ExpertObservation:
    """One explicitly scoped expert invocation."""

    episode_id: str
    step: int
    expert_id: str
    objective: str
    workspace: ScopedWorkspaceView
    local_budget: BudgetLimits | None = None

    def __post_init__(self) -> None:
        require_id(self.episode_id, "episode_id")
        require_step(self.step, "step")
        _require_expert_id(self.expert_id)
        require_text(self.objective, "objective")
        if not isinstance(self.workspace, ScopedWorkspaceView):
            raise TypeError("workspace must be a ScopedWorkspaceView")
        if self.episode_id != self.workspace.episode_id:
            raise ValueError("observation episode_id does not match workspace")
        if self.local_budget is not None and not isinstance(self.local_budget, BudgetLimits):
            raise TypeError("local_budget must be BudgetLimits or None")


@dataclass(frozen=True, slots=True)
class ExpertResult:
    """An expert proposal; routing decisions are deliberately absent."""

    expert_id: str
    workspace_delta: WorkspaceDelta
    capability_request: CapabilityRequest | None = None
    cost_delta: CostVector = field(default_factory=CostVector)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_expert_id(self.expert_id)
        if not isinstance(self.workspace_delta, WorkspaceDelta):
            raise TypeError("workspace_delta must be a WorkspaceDelta")
        if self.workspace_delta.actor_id != self.expert_id:
            raise ValueError("workspace delta actor_id must equal expert_id")
        if self.capability_request is not None:
            if not isinstance(self.capability_request, CapabilityRequest):
                raise TypeError("capability_request must be CapabilityRequest or None")
            if self.capability_request.created_by != self.expert_id:
                raise ValueError("capability request created_by must equal expert_id")
        if not isinstance(self.cost_delta, CostVector):
            raise TypeError("cost_delta must be a CostVector")
        object.__setattr__(self, "metrics", normalize_json_object(self.metrics, "metrics"))

    def validate_against(self, observation: ExpertObservation) -> None:
        if not isinstance(observation, ExpertObservation):
            raise TypeError("observation must be an ExpertObservation")
        if self.expert_id != observation.expert_id:
            raise ValueError("result expert_id does not match observation")
        if self.workspace_delta.episode_id != observation.episode_id:
            raise ValueError("workspace delta episode_id does not match observation")
        if self.workspace_delta.base_version != observation.workspace.version:
            raise ValueError("workspace delta base_version does not match scoped view")
        if (
            self.capability_request is not None
            and self.capability_request.created_step != observation.step
        ):
            raise ValueError("capability request created_step does not match observation")


@runtime_checkable
class CognitiveExpert(Protocol):
    expert_id: str

    def run(self, observation: ExpertObservation) -> ExpertResult:
        ...


__all__ = [
    "CognitiveExpert",
    "EXPERT_IDS",
    "ExpertObservation",
    "ExpertResult",
]
