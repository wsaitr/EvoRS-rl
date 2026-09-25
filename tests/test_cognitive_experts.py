from __future__ import annotations

from dataclasses import fields

import pytest

from rs_hmcomm.contracts import Evidence, RecordStatus, TaskContext
from rs_hmcomm.experts import (
    CognitiveExpert,
    ExpertObservation,
    ExpertResult,
    KnowledgeExpert,
    ReasoningExpert,
    VerificationExpert,
    VisualExpert,
)
from rs_hmcomm.workspace import (
    WorkspaceDelta,
    WorkspaceReducer,
    WorkspaceState,
    build_scoped_view,
)


def initial_state(*, metadata: dict | None = None) -> WorkspaceState:
    return WorkspaceState.initial(
        "episode-experts",
        TaskContext(
            task_id="task-experts",
            image_ref="fixture://airport.png",
            question="How many aircraft are visible at the airport?",
            metadata=metadata or {},
        ),
    )


def observation(
    state: WorkspaceState,
    expert_id: str,
    *,
    step: int,
    visible_record_ids: tuple[str, ...] = (),
) -> ExpertObservation:
    return ExpertObservation(
        episode_id=state.episode_id,
        step=step,
        expert_id=expert_id,
        objective=f"Execute {expert_id} capability",
        workspace=build_scoped_view(state, visible_record_ids),
    )


def test_expert_contract_has_no_routing_or_next_agent_field() -> None:
    assert {item.name for item in fields(ExpertResult)} == {
        "expert_id",
        "workspace_delta",
        "capability_request",
        "cost_delta",
        "metrics",
    }
    for expert in (
        VisualExpert(),
        ReasoningExpert(),
        VerificationExpert(),
        KnowledgeExpert(),
    ):
        assert isinstance(expert, CognitiveExpert)
        assert expert.expert_id in {
            "visual",
            "reasoning",
            "verification",
            "knowledge",
        }


def test_visual_expert_produces_scene_object_and_visual_evidence() -> None:
    state = initial_state()
    obs = observation(state, "visual", step=0)
    result = VisualExpert().run(obs)

    assert result.expert_id == "visual"
    assert result.workspace_delta.base_version == state.version
    assert len(result.workspace_delta.spatial_nodes) == 2
    assert {node.level.value for node in result.workspace_delta.spatial_nodes} == {
        "scene",
        "object",
    }
    assert len(result.workspace_delta.evidence) == 1
    assert result.workspace_delta.evidence[0].kind == "visual"
    assert result.workspace_delta.evidence[0].payload["scientific_claim"] is False

    updated = WorkspaceReducer.apply(state, result.workspace_delta)
    object_node = next(node for node in updated.spatial_nodes if node.level.value == "object")
    assert object_node.parent_id == "visual-scene-s0"
    assert object_node.evidence_ids == ("visual-evidence-s0",)


def test_visual_expert_reuses_visible_scene_instead_of_adding_second_root() -> None:
    state = initial_state()
    first = VisualExpert().run(observation(state, "visual", step=0))
    state = WorkspaceReducer.apply(state, first.workspace_delta)
    second = VisualExpert().run(
        observation(state, "visual", step=1, visible_record_ids=("visual-scene-s0",))
    )

    assert len(second.workspace_delta.spatial_nodes) == 1
    assert second.workspace_delta.spatial_nodes[0].parent_id == "visual-scene-s0"
    WorkspaceReducer.apply(state, second.workspace_delta)


def test_knowledge_expert_produces_domain_knowledge_evidence() -> None:
    state = initial_state()
    result = KnowledgeExpert().run(observation(state, "knowledge", step=0))

    assert result.workspace_delta.base_version == 0
    assert result.workspace_delta.actor_id == "knowledge"
    assert len(result.workspace_delta.evidence) == 1
    evidence = result.workspace_delta.evidence[0]
    assert evidence.kind == "domain_knowledge"
    assert "runways" in evidence.summary
    WorkspaceReducer.apply(state, result.workspace_delta)


def state_with_evidence(*, metadata: dict | None = None) -> WorkspaceState:
    state = initial_state(metadata=metadata)
    delta = WorkspaceDelta(
        delta_id="fixture-evidence-delta",
        episode_id=state.episode_id,
        base_version=0,
        actor_id="visual",
        evidence=(
            Evidence(
                evidence_id="fixture-evidence",
                kind="visual",
                summary="Three distinct aircraft are visible",
                created_by="visual",
            ),
        ),
    )
    return WorkspaceReducer.apply(state, delta)


def test_reasoning_uses_visible_evidence_and_scripted_answer_only_for_fixture() -> None:
    state = state_with_evidence(
        metadata={"scripted_fixture": True, "mock_answer": "3"}
    )
    obs = observation(
        state,
        "reasoning",
        step=1,
        visible_record_ids=("fixture-evidence",),
    )
    result = ReasoningExpert().run(obs)

    assert result.workspace_delta.base_version == state.version
    claim = result.workspace_delta.claims[0]
    candidate = result.workspace_delta.answer_candidates[0]
    assert "non-scientific" in claim.statement
    assert claim.evidence_ids == ("fixture-evidence",)
    assert candidate.answer == "3"
    assert candidate.claim_ids == (claim.claim_id,)
    assert candidate.evidence_ids == ("fixture-evidence",)
    WorkspaceReducer.apply(state, result.workspace_delta)


def test_reasoning_rejects_mock_answer_outside_scripted_fixture() -> None:
    state = state_with_evidence(metadata={"mock_answer": "evaluation-label"})
    obs = observation(
        state,
        "reasoning",
        step=1,
        visible_record_ids=("fixture-evidence",),
    )

    with pytest.raises(ValueError, match="scripted_fixture"):
        ReasoningExpert().run(obs)


def test_reasoning_without_visible_evidence_cannot_make_candidate() -> None:
    state = state_with_evidence(
        metadata={"scripted_fixture": True, "mock_answer": "3"}
    )
    with pytest.raises(ValueError, match="visible evidence"):
        ReasoningExpert().run(observation(state, "reasoning", step=1))


def state_with_reasoning_candidate() -> WorkspaceState:
    state = state_with_evidence(
        metadata={"scripted_fixture": True, "mock_answer": "3"}
    )
    reasoning = ReasoningExpert().run(
        observation(
            state,
            "reasoning",
            step=1,
            visible_record_ids=("fixture-evidence",),
        )
    )
    return WorkspaceReducer.apply(state, reasoning.workspace_delta)


def test_verification_requires_candidate_and_creates_new_verified_candidate() -> None:
    state = state_with_reasoning_candidate()
    obs = observation(
        state,
        "verification",
        step=2,
        visible_record_ids=("reasoning-answer-s1",),
    )
    result = VerificationExpert().run(obs)

    assert result.workspace_delta.base_version == state.version
    assert len(result.workspace_delta.answer_candidates) == 1
    verified = result.workspace_delta.answer_candidates[0]
    original = state.answer_candidates[0]
    assert verified.candidate_id != original.candidate_id
    assert verified.answer == original.answer
    assert verified.claim_ids == original.claim_ids
    assert verified.evidence_ids == original.evidence_ids
    assert verified.status == RecordStatus.VERIFIED
    assert verified.created_by == "verification"

    updated = WorkspaceReducer.apply(state, result.workspace_delta)
    assert len(updated.answer_candidates) == 2


def test_verification_cannot_run_without_visible_candidate() -> None:
    state = state_with_reasoning_candidate()
    with pytest.raises(ValueError, match="visible answer candidate"):
        VerificationExpert().run(observation(state, "verification", step=2))


def test_expert_cannot_execute_observation_assigned_to_another_expert() -> None:
    state = initial_state()
    with pytest.raises(ValueError, match="cannot execute"):
        KnowledgeExpert().run(observation(state, "visual", step=0))
