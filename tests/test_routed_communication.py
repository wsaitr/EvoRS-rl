from __future__ import annotations

import json

import pytest

from rs_hmcomm.artifacts import ArtifactKind, VisualArtifact
from rs_hmcomm.communication import (
    ContextRouter,
    Inbox,
    MessageEnvelope,
    MessageModality,
    ObservationBuilder,
)
from rs_hmcomm.contracts import Claim, Evidence, TaskContext
from rs_hmcomm.workspace import WorkspaceDelta, WorkspaceReducer, WorkspaceState


def workspace() -> WorkspaceState:
    task = TaskContext(
        task_id="task-airport",
        image_ref="fixture://airport.png",
        question="How many aircraft are visible?",
    )
    initial = WorkspaceState.initial("episode-route", task)
    latent = VisualArtifact(
        artifact_id="latent-aircraft",
        kind=ArtifactKind.LATENT,
        media_type="application/x-evors-float32-le",
        shape=(12,),
        dtype="float32",
        byte_size=48,
        source_image_ref=task.image_ref,
        roi=(0.1, 0.1, 0.9, 0.9),
        encoder="pixel_stats_smoke_v1",
        layer="rgb_mean_std_min_max",
        projector="identity_float32",
        compatible_consumers=("reasoning",),
        sha256="a" * 64,
    )
    evidence = Evidence(
        evidence_id="evidence-aircraft",
        kind="visual_count",
        summary="Three aircraft are visible in the crop.",
        created_by="visual",
        artifact_refs=(latent.artifact_id,),
    )
    unrelated = Evidence(
        evidence_id="evidence-hangar",
        kind="visual",
        summary="An unrelated hangar is visible.",
        created_by="visual",
    )
    claim = Claim(
        claim_id="claim-aircraft",
        statement="The scene contains three aircraft.",
        created_by="reasoning",
        evidence_ids=(evidence.evidence_id,),
    )
    return WorkspaceReducer.apply(
        initial,
        WorkspaceDelta(
            delta_id="delta-route-fixture",
            episode_id=initial.episode_id,
            base_version=initial.version,
            actor_id="visual",
            artifacts=(latent,),
            evidence=(evidence, unrelated),
            claims=(claim,),
        ),
    )


def test_route_discloses_only_refs_and_measures_actual_modalities() -> None:
    state = workspace()
    router = ContextRouter()
    envelope, grant = router.route(
        state,
        "visual",
        "reasoning",
        refs=("claim-aircraft", "latent-aircraft"),
        modalities=(
            MessageModality.TEXT,
            MessageModality.STRUCTURE,
            MessageModality.LATENT,
        ),
        step=2,
    )

    serialized = envelope.to_json()
    assert envelope.workspace_refs == ("claim-aircraft", "latent-aircraft")
    assert "Three aircraft are visible in the crop" not in serialized
    assert "The scene contains three aircraft" not in serialized
    assert "a" * 64 not in serialized
    assert grant.message_id == envelope.message_id
    assert grant.grantee == "reasoning"
    assert grant.workspace_refs == envelope.workspace_refs
    assert grant.cost.message_text_tokens > 0
    assert grant.cost.message_structure_bytes == len(serialized.encode("utf-8"))
    assert grant.cost.message_latent_bytes == 48
    assert grant.cost == router.measure_cost(state, envelope)


def test_latent_cost_requires_both_latent_modality_and_artifact_ref() -> None:
    state = workspace()
    router = ContextRouter()
    with pytest.raises(ValueError, match="artifact refs require the latent modality"):
        router.route(
            state,
            "visual",
            "reasoning",
            refs=("latent-aircraft",),
            modalities=(MessageModality.STRUCTURE,),
            step=0,
        )
    no_artifact_ref, no_artifact_grant = router.route(
        state,
        "visual",
        "reasoning",
        refs=("claim-aircraft",),
        modalities=(MessageModality.LATENT,),
        step=1,
    )

    assert no_artifact_grant.cost.message_latent_bytes == 0
    assert no_artifact_ref.modalities == (MessageModality.LATENT,)


def test_recipient_scoping_and_message_removal_change_observation() -> None:
    state = workspace()
    envelope, _ = ContextRouter().route(
        state,
        "visual",
        "reasoning",
        refs=("claim-aircraft",),
        modalities=(MessageModality.STRUCTURE,),
        step=1,
    )
    inbox = Inbox()
    inbox.append(envelope)

    reasoning_view = ObservationBuilder.build(state, "reasoning", inbox)
    verification_view = ObservationBuilder.build(state, "verification", inbox)
    assert reasoning_view.visible_record_ids == {
        "task-airport",
        "claim-aircraft",
        "evidence-aircraft",
    }
    assert reasoning_view.evidence[0].artifact_refs == ()
    assert verification_view.visible_record_ids == {"task-airport"}
    assert "evidence-hangar" not in reasoning_view.visible_record_ids

    assert inbox.remove(envelope.message_id) == envelope
    after_removal = ObservationBuilder.build(state, "reasoning", inbox)
    assert after_removal.visible_record_ids == {"task-airport"}


def test_latent_disclosure_requires_compatible_recipient_and_is_visible_when_paid() -> None:
    state = workspace()
    router = ContextRouter()
    with pytest.raises(PermissionError, match="not compatible"):
        router.route(
            state,
            "visual",
            "verification",
            refs=("latent-aircraft",),
            modalities=(MessageModality.LATENT,),
            step=0,
        )

    envelope, grant = router.route(
        state,
        "visual",
        "reasoning",
        refs=("claim-aircraft", "latent-aircraft"),
        modalities=(MessageModality.STRUCTURE, MessageModality.LATENT),
        step=1,
    )
    inbox = Inbox()
    inbox.append(envelope)
    view = ObservationBuilder.build(state, "reasoning", inbox)
    assert "latent-aircraft" in view.visible_record_ids
    assert view.evidence[0].artifact_refs == ("latent-aircraft",)
    assert grant.cost.message_latent_bytes == 48


def test_inbox_filters_replays_and_rejects_duplicate_messages() -> None:
    state = workspace()
    router = ContextRouter()
    reasoning, _ = router.route(
        state,
        "visual",
        "reasoning",
        refs=("evidence-aircraft",),
        modalities=(MessageModality.TEXT,),
        step=0,
    )
    verification, _ = router.route(
        state,
        "visual",
        "verification",
        refs=("evidence-hangar",),
        modalities=(MessageModality.TEXT,),
        step=0,
    )
    inbox = Inbox()
    inbox.append(reasoning)
    inbox.append(verification)

    assert inbox.by_recipient("reasoning") == (reasoning,)
    assert inbox.by_recipient("verification") == (verification,)
    with pytest.raises(ValueError, match="duplicate message_id"):
        inbox.append(reasoning)

    serialized_log = json.loads(json.dumps(inbox.to_dict()))
    replayed = Inbox.replay(serialized_log)
    assert replayed.messages == inbox.messages
    assert MessageEnvelope.from_json(reasoning.to_json()) == reasoning


def test_router_rejects_unknown_workspace_refs_before_issuing_grant() -> None:
    with pytest.raises(ValueError, match="unknown workspace refs"):
        ContextRouter().route(
            workspace(),
            "visual",
            "reasoning",
            refs=("missing-record",),
            modalities=(MessageModality.TEXT,),
            step=0,
        )


def test_message_contract_rejects_duplicate_or_empty_disclosures() -> None:
    common = {
        "message_id": "message-invalid",
        "sender": "visual",
        "receiver": "reasoning",
        "workspace_refs": ("evidence-aircraft",),
        "scope": "workspace_subset",
        "created_step": 0,
    }
    with pytest.raises(ValueError, match="duplicates"):
        MessageEnvelope(
            modalities=(MessageModality.TEXT, MessageModality.TEXT),
            **common,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        MessageEnvelope(
            modalities=(MessageModality.TEXT,),
            **{**common, "workspace_refs": ()},
        )
