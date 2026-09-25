"""The only supported mutation path for the immutable shared workspace."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Iterable

from rs_hmcomm.artifacts import VisualArtifact
from rs_hmcomm.contracts import AnswerCandidate, Claim, Evidence
from rs_hmcomm.spatial import SpatialNode

from .state import ScopedWorkspaceView, WorkspaceDelta, WorkspaceState


class WorkspaceConflictError(ValueError):
    """Raised when a delta targets the wrong episode or base version."""


class WorkspaceReducer:
    """Apply append-only deltas and produce a new immutable snapshot."""

    @staticmethod
    def apply(state: WorkspaceState, delta: WorkspaceDelta) -> WorkspaceState:
        if not isinstance(state, WorkspaceState):
            raise TypeError("state must be a WorkspaceState")
        if not isinstance(delta, WorkspaceDelta):
            raise TypeError("delta must be a WorkspaceDelta")
        if delta.episode_id != state.episode_id:
            raise WorkspaceConflictError(
                f"delta episode {delta.episode_id} does not match {state.episode_id}"
            )
        if delta.base_version != state.version:
            raise WorkspaceConflictError(
                f"delta base_version {delta.base_version} does not match "
                f"workspace version {state.version}"
            )

        # WorkspaceState validation is intentionally the final integrity gate:
        # additions in one delta may reference each other, but no dangling or
        # duplicate record can survive into the returned snapshot.
        return WorkspaceState(
            episode_id=state.episode_id,
            version=state.version + 1,
            task=state.task,
            evidence=state.evidence + delta.evidence,
            claims=state.claims + delta.claims,
            answer_candidates=state.answer_candidates + delta.answer_candidates,
            spatial_nodes=state.spatial_nodes + delta.spatial_nodes,
            artifacts=state.artifacts + delta.artifacts,
        )


def _record_map(state: WorkspaceState) -> dict[str, object]:
    records: dict[str, object] = {state.task.task_id: state.task}
    records.update((item.evidence_id, item) for item in state.evidence)
    records.update((item.claim_id, item) for item in state.claims)
    records.update((item.candidate_id, item) for item in state.answer_candidates)
    records.update((item.node_id, item) for item in state.spatial_nodes)
    records.update((item.artifact_id, item) for item in state.artifacts)
    return records


def _dependencies(record: object, artifact_ids: frozenset[str]) -> tuple[str, ...]:
    if isinstance(record, Evidence):
        return (*record.spatial_node_ids, *record.artifact_refs)
    if isinstance(record, Claim):
        return record.evidence_ids
    if isinstance(record, AnswerCandidate):
        return (*record.claim_ids, *record.evidence_ids)
    if isinstance(record, SpatialNode):
        parent = (record.parent_id,) if record.parent_id is not None else ()
        return (*parent, *record.evidence_ids, *record.visual_artifact_refs)
    if isinstance(record, VisualArtifact):
        # External task image refs are not workspace records.  Derived
        # artifact-to-artifact provenance is dependency-closed when present.
        if record.source_image_ref in artifact_ids:
            return (record.source_image_ref,)
    return ()


def build_scoped_view(
    state: WorkspaceState,
    visible_record_ids: Iterable[str],
    *,
    allowed_artifact_ids: Iterable[str] | None = None,
) -> ScopedWorkspaceView:
    """Build the smallest dependency-closed view allowed by explicit grants.

    The task is always visible.  Granting an answer, claim, evidence, spatial
    node, or artifact recursively grants only the records that it references.
    Unrelated records remain absent.
    """

    if not isinstance(state, WorkspaceState):
        raise TypeError("state must be a WorkspaceState")
    if isinstance(visible_record_ids, (str, bytes)):
        raise TypeError("visible_record_ids must be an iterable of record IDs")
    requested = tuple(visible_record_ids)
    if any(not isinstance(item, str) or not item.strip() for item in requested):
        raise ValueError("visible_record_ids must contain non-empty strings")

    records = _record_map(state)
    unknown = set(requested) - set(records)
    if unknown:
        raise ValueError(f"scoped view contains unknown record IDs: {sorted(unknown)}")

    artifact_ids = frozenset(item.artifact_id for item in state.artifacts)
    if allowed_artifact_ids is None:
        allowed_artifacts = artifact_ids
    else:
        if isinstance(allowed_artifact_ids, (str, bytes)):
            raise TypeError("allowed_artifact_ids must be an iterable of artifact IDs")
        allowed_artifacts = frozenset(allowed_artifact_ids)
        unknown_artifacts = allowed_artifacts - artifact_ids
        if unknown_artifacts:
            raise ValueError(
                "allowed_artifact_ids contains unknown artifacts: "
                f"{sorted(unknown_artifacts)}"
            )
        disallowed_requests = set(requested) & (artifact_ids - allowed_artifacts)
        if disallowed_requests:
            raise ValueError(
                "visible_record_ids directly requests artifacts without access: "
                f"{sorted(disallowed_requests)}"
            )
    closure = {state.task.task_id, *requested}
    queue = deque(requested)
    while queue:
        record_id = queue.popleft()
        record = records[record_id]
        for dependency_id in _dependencies(record, artifact_ids):
            if dependency_id in artifact_ids and dependency_id not in allowed_artifacts:
                continue
            if dependency_id not in closure:
                closure.add(dependency_id)
                queue.append(dependency_id)

    # Do not leave a usable artifact handle in a view that was not granted the
    # corresponding artifact record. This keeps structure-only disclosure from
    # becoming a zero-cost side channel into ArtifactStore.
    evidence = tuple(
        replace(
            item,
            artifact_refs=tuple(ref for ref in item.artifact_refs if ref in closure),
        )
        for item in state.evidence
        if item.evidence_id in closure
    )
    spatial_nodes = tuple(
        replace(
            item,
            visual_artifact_refs=tuple(
                ref for ref in item.visual_artifact_refs if ref in closure
            ),
        )
        for item in state.spatial_nodes
        if item.node_id in closure
    )

    return ScopedWorkspaceView(
        episode_id=state.episode_id,
        version=state.version,
        task=state.task,
        evidence=evidence,
        claims=tuple(item for item in state.claims if item.claim_id in closure),
        answer_candidates=tuple(
            item for item in state.answer_candidates if item.candidate_id in closure
        ),
        spatial_nodes=spatial_nodes,
        artifacts=tuple(item for item in state.artifacts if item.artifact_id in closure),
    )


__all__ = ["WorkspaceConflictError", "WorkspaceReducer", "build_scoped_view"]
