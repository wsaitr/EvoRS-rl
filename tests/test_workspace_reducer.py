from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rs_hmcomm.artifacts import VisualArtifact
from rs_hmcomm.contracts import AnswerCandidate, Claim, Evidence, TaskContext
from rs_hmcomm.spatial import SpatialNode
from rs_hmcomm.workspace import (
    WorkspaceConflictError,
    WorkspaceDelta,
    WorkspaceIntegrityError,
    WorkspaceReducer,
    WorkspaceState,
    build_scoped_view,
)


def task() -> TaskContext:
    return TaskContext(
        task_id="task-airport",
        image_ref="fixture://airport.png",
        question="How many aircraft are visible?",
    )


def artifact(
    artifact_id: str,
    *,
    source_image_ref: str = "fixture://airport.png",
    kind: str = "image",
) -> VisualArtifact:
    return VisualArtifact(
        artifact_id=artifact_id,
        kind=kind,
        media_type="image/png",
        shape=(32, 32, 3),
        dtype="uint8",
        byte_size=3072,
        source_image_ref=source_image_ref,
        roi=None,
        encoder=None,
        layer=None,
        projector=None,
        compatible_consumers=(),
        sha256="a" * 64,
    )


def complete_first_delta() -> WorkspaceDelta:
    evidence = Evidence(
        evidence_id="evidence-1",
        kind="visual_count",
        summary="Three aircraft are visible",
        created_by="generalist",
        spatial_node_ids=("object-1",),
        artifact_refs=("artifact-1",),
    )
    return WorkspaceDelta(
        delta_id="delta-1",
        episode_id="episode-1",
        base_version=0,
        actor_id="generalist",
        artifacts=(artifact("artifact-1"),),
        spatial_nodes=(
            SpatialNode(
                node_id="scene-1",
                level="scene",
                semantic="airport scene",
            ),
            SpatialNode(
                node_id="object-1",
                level="object",
                semantic="aircraft group",
                bbox=(0.1, 0.1, 0.5, 0.5),
                parent_id="scene-1",
                evidence_ids=("evidence-1",),
                visual_artifact_refs=("artifact-1",),
            ),
        ),
        evidence=(evidence,),
        claims=(
            Claim(
                claim_id="claim-1",
                statement="The image contains three aircraft",
                created_by="generalist",
                evidence_ids=("evidence-1",),
            ),
        ),
        answer_candidates=(
            AnswerCandidate(
                candidate_id="answer-1",
                answer="3",
                created_by="generalist",
                claim_ids=("claim-1",),
                evidence_ids=("evidence-1",),
            ),
        ),
    )


def test_reducer_is_append_only_and_returns_a_new_version() -> None:
    initial = WorkspaceState.initial("episode-1", task())
    updated = WorkspaceReducer.apply(initial, complete_first_delta())

    assert initial.version == 0
    assert initial.evidence == ()
    assert updated.version == 1
    assert updated.episode_id == initial.episode_id
    assert updated.task is initial.task
    assert updated.record_ids == {
        "task-airport",
        "artifact-1",
        "scene-1",
        "object-1",
        "evidence-1",
        "claim-1",
        "answer-1",
    }
    with pytest.raises(FrozenInstanceError):
        updated.version = 2  # type: ignore[misc]


def test_same_delta_can_add_mutually_referenced_records_and_artifact_chain() -> None:
    initial = WorkspaceState.initial("episode-1", task())
    base = artifact("artifact-base")
    crop = artifact(
        "artifact-crop",
        source_image_ref="artifact-base",
        kind="crop",
    )
    evidence = Evidence(
        evidence_id="evidence-circular",
        kind="crop",
        summary="ROI evidence",
        created_by="generalist",
        spatial_node_ids=("object-circular",),
        artifact_refs=("artifact-crop",),
    )
    delta = WorkspaceDelta(
        delta_id="delta-circular",
        episode_id="episode-1",
        base_version=0,
        actor_id="generalist",
        artifacts=(base, crop),
        evidence=(evidence,),
        spatial_nodes=(
            SpatialNode("scene-circular", "scene", "scene"),
            SpatialNode(
                "object-circular",
                "object",
                "aircraft",
                parent_id="scene-circular",
                evidence_ids=("evidence-circular",),
                visual_artifact_refs=("artifact-crop",),
            ),
        ),
    )

    state = WorkspaceReducer.apply(initial, delta)
    assert state.version == 1
    assert state.artifacts[1].source_image_ref == "artifact-base"


def test_reducer_rejects_wrong_episode_and_stale_version() -> None:
    initial = WorkspaceState.initial("episode-1", task())
    wrong_episode = WorkspaceDelta(
        delta_id="wrong-episode",
        episode_id="episode-2",
        base_version=0,
        actor_id="generalist",
        artifacts=(artifact("artifact-wrong"),),
    )
    stale = WorkspaceDelta(
        delta_id="stale",
        episode_id="episode-1",
        base_version=1,
        actor_id="generalist",
        artifacts=(artifact("artifact-stale"),),
    )

    with pytest.raises(WorkspaceConflictError, match="episode"):
        WorkspaceReducer.apply(initial, wrong_episode)
    with pytest.raises(WorkspaceConflictError, match="base_version"):
        WorkspaceReducer.apply(initial, stale)


def test_cross_record_duplicate_id_is_rejected_without_mutating_state() -> None:
    state = WorkspaceReducer.apply(
        WorkspaceState.initial("episode-1", task()),
        complete_first_delta(),
    )
    duplicate = WorkspaceDelta(
        delta_id="delta-duplicate",
        episode_id="episode-1",
        base_version=1,
        actor_id="generalist",
        evidence=(
            Evidence(
                evidence_id="claim-1",
                kind="duplicate",
                summary="must fail",
                created_by="generalist",
            ),
        ),
    )

    with pytest.raises(WorkspaceIntegrityError, match="duplicate workspace record ID"):
        WorkspaceReducer.apply(state, duplicate)
    assert state.version == 1
    assert len(state.evidence) == 1


def invalid_reference_deltas() -> tuple[WorkspaceDelta, ...]:
    common = {
        "episode_id": "episode-1",
        "base_version": 0,
        "actor_id": "generalist",
    }
    return (
        WorkspaceDelta(
            delta_id="dangling-evidence-spatial",
            evidence=(
                Evidence(
                    "evidence-dangling-spatial",
                    "visual",
                    "bad",
                    "generalist",
                    spatial_node_ids=("missing-node",),
                ),
            ),
            **common,
        ),
        WorkspaceDelta(
            delta_id="dangling-evidence-artifact",
            evidence=(
                Evidence(
                    "evidence-dangling-artifact",
                    "visual",
                    "bad",
                    "generalist",
                    artifact_refs=("missing-artifact",),
                ),
            ),
            **common,
        ),
        WorkspaceDelta(
            delta_id="dangling-claim",
            claims=(
                Claim(
                    "claim-dangling",
                    "bad",
                    "generalist",
                    ("missing-evidence",),
                ),
            ),
            **common,
        ),
        WorkspaceDelta(
            delta_id="dangling-answer",
            answer_candidates=(
                AnswerCandidate(
                    "answer-dangling",
                    "bad",
                    "generalist",
                    claim_ids=("missing-claim",),
                ),
            ),
            **common,
        ),
        WorkspaceDelta(
            delta_id="dangling-node-evidence",
            spatial_nodes=(
                SpatialNode(
                    "scene-dangling-evidence",
                    "scene",
                    "scene",
                    evidence_ids=("missing-evidence",),
                ),
            ),
            **common,
        ),
        WorkspaceDelta(
            delta_id="dangling-node-artifact",
            spatial_nodes=(
                SpatialNode(
                    "scene-dangling-artifact",
                    "scene",
                    "scene",
                    visual_artifact_refs=("missing-artifact",),
                ),
            ),
            **common,
        ),
        WorkspaceDelta(
            delta_id="dangling-artifact-source",
            artifacts=(
                artifact(
                    "artifact-dangling-source",
                    source_image_ref="missing-source",
                ),
            ),
            **common,
        ),
    )


@pytest.mark.parametrize("delta", invalid_reference_deltas())
def test_every_workspace_reference_must_resolve(delta: WorkspaceDelta) -> None:
    initial = WorkspaceState.initial("episode-1", task())
    with pytest.raises(WorkspaceIntegrityError, match="dangling"):
        WorkspaceReducer.apply(initial, delta)


def test_spatial_graph_validation_runs_during_apply() -> None:
    delta = WorkspaceDelta(
        delta_id="invalid-hierarchy",
        episode_id="episode-1",
        base_version=0,
        actor_id="generalist",
        spatial_nodes=(
            SpatialNode(
                "object-with-missing-parent",
                "object",
                "aircraft",
                parent_id="missing-scene",
            ),
        ),
    )

    with pytest.raises(WorkspaceIntegrityError, match="invalid spatial hierarchy"):
        WorkspaceReducer.apply(WorkspaceState.initial("episode-1", task()), delta)


def two_branch_state() -> WorkspaceState:
    state = WorkspaceReducer.apply(
        WorkspaceState.initial("episode-1", task()),
        complete_first_delta(),
    )
    evidence = Evidence(
        "evidence-2",
        "visual_count",
        "Unrelated second observation",
        "generalist",
        spatial_node_ids=("object-2",),
        artifact_refs=("artifact-2",),
    )
    return WorkspaceReducer.apply(
        state,
        WorkspaceDelta(
            delta_id="delta-2",
            episode_id="episode-1",
            base_version=1,
            actor_id="generalist",
            artifacts=(artifact("artifact-2"),),
            spatial_nodes=(
                SpatialNode(
                    "object-2",
                    "object",
                    "hangar",
                    parent_id="scene-1",
                    evidence_ids=("evidence-2",),
                    visual_artifact_refs=("artifact-2",),
                ),
            ),
            evidence=(evidence,),
            claims=(Claim("claim-2", "A hangar exists", "generalist", ("evidence-2",)),),
            answer_candidates=(
                AnswerCandidate(
                    "answer-2",
                    "hangar",
                    "generalist",
                    claim_ids=("claim-2",),
                ),
            ),
        ),
    )


def test_scoped_view_includes_only_grant_and_dependency_closure() -> None:
    state = two_branch_state()
    view = build_scoped_view(state, {"answer-1"})

    assert view.task is state.task
    assert view.visible_record_ids == {
        "task-airport",
        "answer-1",
        "claim-1",
        "evidence-1",
        "object-1",
        "scene-1",
        "artifact-1",
    }
    assert not (
        {"answer-2", "claim-2", "evidence-2", "object-2", "artifact-2"}
        & view.visible_record_ids
    )


def test_task_is_always_visible_and_unknown_grants_are_rejected() -> None:
    state = two_branch_state()
    task_only = build_scoped_view(state, ())

    assert task_only.visible_record_ids == {"task-airport"}
    assert task_only.evidence == ()
    assert task_only.spatial_nodes == ()
    with pytest.raises(ValueError, match="unknown record IDs"):
        build_scoped_view(state, {"not-a-record"})


def test_artifact_grant_recursively_includes_its_source_descriptor() -> None:
    initial = WorkspaceState.initial("episode-1", task())
    state = WorkspaceReducer.apply(
        initial,
        WorkspaceDelta(
            delta_id="artifact-chain",
            episode_id="episode-1",
            base_version=0,
            actor_id="generalist",
            artifacts=(
                artifact("artifact-source"),
                artifact(
                    "artifact-derived",
                    source_image_ref="artifact-source",
                    kind="crop",
                ),
            ),
        ),
    )

    view = build_scoped_view(state, {"artifact-derived"})
    assert view.visible_record_ids == {
        "task-airport",
        "artifact-source",
        "artifact-derived",
    }
