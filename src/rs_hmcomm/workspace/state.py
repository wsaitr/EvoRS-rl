"""Immutable, episode-scoped shared workspace snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rs_hmcomm.artifacts import VisualArtifact
from rs_hmcomm.contracts import AnswerCandidate, Claim, Evidence, TaskContext
from rs_hmcomm.contracts._validation import require_id
from rs_hmcomm.spatial import SpatialNode, validate_spatial_hierarchy


class WorkspaceIntegrityError(ValueError):
    """Raised when a workspace snapshot contains invalid records or refs."""


def _non_negative_version(value: int, field_name: str = "version") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _record_entries(
    task: TaskContext,
    evidence: tuple[Evidence, ...],
    claims: tuple[Claim, ...],
    answer_candidates: tuple[AnswerCandidate, ...],
    spatial_nodes: tuple[SpatialNode, ...],
    artifacts: tuple[VisualArtifact, ...],
) -> tuple[tuple[str, str], ...]:
    return (
        (task.task_id, "task"),
        *((item.evidence_id, "evidence") for item in evidence),
        *((item.claim_id, "claim") for item in claims),
        *((item.candidate_id, "answer_candidate") for item in answer_candidates),
        *((item.node_id, "spatial_node") for item in spatial_nodes),
        *((item.artifact_id, "artifact") for item in artifacts),
    )


def _require_unique_record_ids(entries: Iterable[tuple[str, str]]) -> None:
    owners: dict[str, str] = {}
    for record_id, kind in entries:
        previous = owners.get(record_id)
        if previous is not None:
            raise WorkspaceIntegrityError(
                f"duplicate workspace record ID {record_id}: {previous} and {kind}"
            )
        owners[record_id] = kind


def _require_refs(
    refs: Iterable[str],
    allowed: frozenset[str],
    *,
    owner: str,
    relation: str,
) -> None:
    missing = set(refs) - allowed
    if missing:
        raise WorkspaceIntegrityError(
            f"{owner} has dangling {relation} refs: {sorted(missing)}"
        )


def _validate_workspace_records(
    task: TaskContext,
    evidence: tuple[Evidence, ...],
    claims: tuple[Claim, ...],
    answer_candidates: tuple[AnswerCandidate, ...],
    spatial_nodes: tuple[SpatialNode, ...],
    artifacts: tuple[VisualArtifact, ...],
) -> None:
    _require_unique_record_ids(
        _record_entries(
            task,
            evidence,
            claims,
            answer_candidates,
            spatial_nodes,
            artifacts,
        )
    )

    evidence_ids = frozenset(item.evidence_id for item in evidence)
    claim_ids = frozenset(item.claim_id for item in claims)
    spatial_node_ids = frozenset(item.node_id for item in spatial_nodes)
    artifact_ids = frozenset(item.artifact_id for item in artifacts)

    # An empty hierarchy is valid for the version-zero workspace.  As soon as
    # one node is committed, the spatial module requires one complete Scene
    # root and validates all parent edges and levels.
    if spatial_nodes:
        try:
            validate_spatial_hierarchy(spatial_nodes)
        except (TypeError, ValueError) as exc:
            raise WorkspaceIntegrityError(f"invalid spatial hierarchy: {exc}") from exc

    for item in evidence:
        _require_refs(
            item.spatial_node_ids,
            spatial_node_ids,
            owner=f"evidence {item.evidence_id}",
            relation="spatial node",
        )
        _require_refs(
            item.artifact_refs,
            artifact_ids,
            owner=f"evidence {item.evidence_id}",
            relation="artifact",
        )

    for item in claims:
        _require_refs(
            item.evidence_ids,
            evidence_ids,
            owner=f"claim {item.claim_id}",
            relation="evidence",
        )

    for item in answer_candidates:
        _require_refs(
            item.claim_ids,
            claim_ids,
            owner=f"answer candidate {item.candidate_id}",
            relation="claim",
        )
        _require_refs(
            item.evidence_ids,
            evidence_ids,
            owner=f"answer candidate {item.candidate_id}",
            relation="evidence",
        )

    for item in spatial_nodes:
        _require_refs(
            item.evidence_ids,
            evidence_ids,
            owner=f"spatial node {item.node_id}",
            relation="evidence",
        )
        _require_refs(
            item.visual_artifact_refs,
            artifact_ids,
            owner=f"spatial node {item.node_id}",
            relation="artifact",
        )

    # A descriptor must ultimately originate from the task image or another
    # descriptor already present in this same immutable snapshot.  This keeps
    # crop/latent provenance closed without embedding physical data here.
    allowed_sources = artifact_ids | frozenset({task.image_ref})
    for item in artifacts:
        if item.source_image_ref not in allowed_sources:
            raise WorkspaceIntegrityError(
                f"artifact {item.artifact_id} has dangling source_image_ref "
                f"{item.source_image_ref}"
            )


@dataclass(frozen=True, slots=True)
class WorkspaceState:
    """One immutable shared-state version for a single episode."""

    episode_id: str
    version: int
    task: TaskContext
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    answer_candidates: tuple[AnswerCandidate, ...] = ()
    spatial_nodes: tuple[SpatialNode, ...] = ()
    artifacts: tuple[VisualArtifact, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.episode_id, "episode_id")
        _non_negative_version(self.version)
        if not isinstance(self.task, TaskContext):
            raise TypeError("task must be a TaskContext")
        expected_types = (
            ("evidence", Evidence),
            ("claims", Claim),
            ("answer_candidates", AnswerCandidate),
            ("spatial_nodes", SpatialNode),
            ("artifacts", VisualArtifact),
        )
        for field_name, record_type in expected_types:
            normalized = tuple(getattr(self, field_name))
            if any(not isinstance(item, record_type) for item in normalized):
                raise TypeError(f"{field_name} must contain {record_type.__name__} records")
            object.__setattr__(self, field_name, normalized)
        _validate_workspace_records(
            self.task,
            self.evidence,
            self.claims,
            self.answer_candidates,
            self.spatial_nodes,
            self.artifacts,
        )

    @classmethod
    def initial(cls, episode_id: str, task: TaskContext) -> "WorkspaceState":
        return cls(episode_id=episode_id, version=0, task=task)

    @property
    def record_ids(self) -> frozenset[str]:
        return frozenset(
            record_id
            for record_id, _ in _record_entries(
                self.task,
                self.evidence,
                self.claims,
                self.answer_candidates,
                self.spatial_nodes,
                self.artifacts,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "version": self.version,
            "task": self.task.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "answer_candidates": [item.to_dict() for item in self.answer_candidates],
            "spatial_nodes": [item.to_dict() for item in self.spatial_nodes],
            "artifacts": [item.descriptor() for item in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class WorkspaceDelta:
    """Append-only records based on one exact workspace version."""

    delta_id: str
    episode_id: str
    base_version: int
    actor_id: str
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    answer_candidates: tuple[AnswerCandidate, ...] = ()
    spatial_nodes: tuple[SpatialNode, ...] = ()
    artifacts: tuple[VisualArtifact, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.delta_id, "delta_id")
        require_id(self.episode_id, "episode_id")
        require_id(self.actor_id, "actor_id")
        _non_negative_version(self.base_version, "base_version")
        expected_types = (
            ("evidence", Evidence),
            ("claims", Claim),
            ("answer_candidates", AnswerCandidate),
            ("spatial_nodes", SpatialNode),
            ("artifacts", VisualArtifact),
        )
        addition_count = 0
        for field_name, record_type in expected_types:
            normalized = tuple(getattr(self, field_name))
            if any(not isinstance(item, record_type) for item in normalized):
                raise TypeError(f"{field_name} must contain {record_type.__name__} records")
            object.__setattr__(self, field_name, normalized)
            addition_count += len(normalized)
        if addition_count == 0:
            raise ValueError("WorkspaceDelta must append at least one record")

    @property
    def addition_count(self) -> int:
        return sum(
            len(items)
            for items in (
                self.evidence,
                self.claims,
                self.answer_candidates,
                self.spatial_nodes,
                self.artifacts,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "episode_id": self.episode_id,
            "base_version": self.base_version,
            "actor_id": self.actor_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "answer_candidates": [item.to_dict() for item in self.answer_candidates],
            "spatial_nodes": [item.to_dict() for item in self.spatial_nodes],
            "artifacts": [item.descriptor() for item in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class ScopedWorkspaceView:
    """Read-only dependency-closed subset produced from an authorized ID set."""

    episode_id: str
    version: int
    task: TaskContext
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    answer_candidates: tuple[AnswerCandidate, ...] = ()
    spatial_nodes: tuple[SpatialNode, ...] = ()
    artifacts: tuple[VisualArtifact, ...] = ()

    @property
    def visible_record_ids(self) -> frozenset[str]:
        return frozenset(
            {
                self.task.task_id,
                *(item.evidence_id for item in self.evidence),
                *(item.claim_id for item in self.claims),
                *(item.candidate_id for item in self.answer_candidates),
                *(item.node_id for item in self.spatial_nodes),
                *(item.artifact_id for item in self.artifacts),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "version": self.version,
            "task": self.task.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "answer_candidates": [item.to_dict() for item in self.answer_candidates],
            "spatial_nodes": [item.to_dict() for item in self.spatial_nodes],
            "artifacts": [item.descriptor() for item in self.artifacts],
        }


__all__ = [
    "ScopedWorkspaceView",
    "WorkspaceDelta",
    "WorkspaceIntegrityError",
    "WorkspaceState",
]
