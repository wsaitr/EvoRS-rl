"""Unit tests for the VLM-backed expert mechanisms (offline, fake backend)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from rs_hmcomm.backends.base import AgentOutput, BackendUsage, TokenAccounting
from rs_hmcomm.contracts import TaskContext
from rs_hmcomm.experts import (
    ExpertObservation,
    ImageStore,
    VLMReasoningExpert,
    VLMVerificationExpert,
    VLMVisualExpert,
)
from rs_hmcomm.workspace import WorkspaceState, build_scoped_view


class FakeBackend:
    """Scripted backend returning queued texts with measured usage."""

    model_name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[BackendRequest] = []

    def generate_turn(self, request: Any) -> AgentOutput:
        from rs_hmcomm.backends.base import BackendRequest

        if not isinstance(request, BackendRequest):
            raise TypeError("expected BackendRequest")
        self.calls.append(request)
        text = self._responses.pop(0)
        return AgentOutput(
            text=text,
            usage=BackendUsage(
                input_tokens=100,
                output_tokens=10,
                token_accounting=TokenAccounting.MEASURED,
            ),
        )


def _store_with_image():
    store = ImageStore()
    store.put("img://x", object())
    return store


def _make_observation(expert_id: str, responses_for_backend: list[str], *, image=None):
    backend = FakeBackend(responses_for_backend)
    store = ImageStore()
    store.put("img://x", image or object())
    task = TaskContext(
        task_id="t",
        image_ref="img://x",
        question="How many bridges?",
        task_type="object_quantity",
        answer_format="short_text",
    )
    state = WorkspaceState.initial("ep", task)
    view = build_scoped_view(state, frozenset())
    observation = ExpertObservation(
        episode_id="ep",
        step=0,
        expert_id=expert_id,
        objective="test",
        workspace=view,
    )
    return backend, observation


class TestBboxParsing:
    def test_pixel_coordinates_normalized_on_1024_canvas(self):
        from rs_hmcomm.experts.vlm_mechanisms import _parse_bboxes

        boxes = _parse_bboxes("[213, 703, 278, 778]")
        assert boxes == [(213 / 1024, 703 / 1024, 278 / 1024, 778 / 1024)]

    def test_normalized_passthrough(self):
        from rs_hmcomm.experts.vlm_mechanisms import _parse_bboxes

        assert _parse_bboxes("[0.1, 0.2, 0.3, 0.4]") == [(0.1, 0.2, 0.3, 0.4)]

    def test_degenerate_boxes_rejected(self):
        from rs_hmcomm.experts.vlm_mechanisms import _parse_bboxes

        assert _parse_bboxes("[0.5, 0.5, 0.5, 0.5]") == []
        assert _parse_bboxes("[0.9, 0.2, 0.1, 0.4]") == []


class TestVLMVisualExpert:
    def test_detections_become_object_nodes_under_scene(self):
        response = json.dumps(
            {
                "objects": [
                    {"label": "bridge", "bbox": [200, 700, 280, 780]},
                    {"label": "tree", "bbox": [50, 60, 90, 110]},
                ]
            }
        )
        backend, observation = _make_observation("visual", [response])
        result = VLMVisualExpert(backend, image_store=_store_with_image()).run(observation)
        nodes = result.workspace_delta.spatial_nodes
        levels = [node.level.value for node in nodes]
        assert levels.count("scene") == 1
        assert levels.count("object") == 2
        scene = next(n for n in nodes if n.level.value == "scene")
        for node in nodes:
            if node.level.value == "object":
                assert node.parent_id == scene.node_id
        evidence = result.workspace_delta.evidence[0]
        assert evidence.payload["model_used"] is True
        assert len(backend.calls) == 1

    def test_unparsable_response_falls_back_to_whole_image(self):
        backend, observation = _make_observation("visual", ["I see no objects."])
        result = VLMVisualExpert(backend, image_store=_store_with_image()).run(observation)
        objects = [
            n for n in result.workspace_delta.spatial_nodes if n.level.value == "object"
        ]
        assert len(objects) == 1
        assert objects[0].bbox == (0.0, 0.0, 1.0, 1.0)
        assert objects[0].provenance["source"] == "vlm_fallback_whole_image"

    def test_bbox_2d_key_variant_parsed(self):
        response = '[{"bbox_2d": [100, 100, 300, 300], "label": "square"}]'
        backend, observation = _make_observation(
            "visual",
            [json.dumps({"objects": [{"label": "square", "bbox": [100, 100, 300, 300]}]})],
        )
        result = VLMVisualExpert(backend, image_store=_store_with_image()).run(observation)
        obj = next(
            n for n in result.workspace_delta.spatial_nodes if n.level.value == "object"
        )
        expected = tuple(v / 1024 for v in (100, 100, 300, 300))
        assert all(abs(a - b) < 1e-6 for a, b in zip(obj.bbox, expected))


class TestLatentConsumption:
    def test_reasoning_consumes_granted_latents_into_prompt(self):
        class FakeConsumer:
            def __init__(self, store, consumer_id):
                pass

            def consume(self, artifact_id):
                return (1.0,) * 8

        latent_object = SimpleNamespace(
            artifact_id="latent-object-0-s0",
            kind=SimpleNamespace(value="latent"),
            compatible_consumers=("reasoning",),
        )
        latent_scene = SimpleNamespace(
            artifact_id="latent-scene-s0",
            kind=SimpleNamespace(value="latent"),
            compatible_consumers=("reasoning",),
        )
        evidence = SimpleNamespace(evidence_id="visual-evidence-s0")

        expert = VLMReasoningExpert(
            FakeBackend(["2"]),
            image_store=ImageStore(),
            artifact_store=SimpleNamespace(),
            latent_consumer_factory=FakeConsumer,
        )
        workspace = SimpleNamespace(
            task=TaskContext(
                task_id="t",
                image_ref="missing://x",
                question="How many?",
                task_type="object_quantity",
                answer_format="short_text",
            ),
            evidence=(evidence,),
            spatial_nodes=(),
            artifacts=(latent_scene, latent_object),
            version=1,
            episode_id="ep",
        )
        diagnostics = expert._consume_visible_latents(workspace)
        assert "latent-scene-s0" in diagnostics
        assert any(key.endswith("~scene_cos") for key in diagnostics)

    def test_no_store_means_no_consumption(self):
        expert = VLMReasoningExpert(FakeBackend(["2"]))
        workspace = SimpleNamespace(artifacts=())
        assert expert._consume_visible_latents(workspace) == {}


class TestVLMVerificationExpert:
    def test_reject_revises_answer(self):
        source_candidate = SimpleNamespace(
            candidate_id="reasoning-answer-s0",
            answer="5",
            claim_ids=("claim-1",),
            evidence_ids=("ev-1",),
            confidence=0.6,
        )
        claim = SimpleNamespace(claim_id="claim-1")

        class ViewWithCandidate(SimpleNamespace):
            pass

        backend = FakeBackend(['{"verdict": "reject", "revised_answer": "2"}'])
        store_img = ImageStore()
        store_img.put("img://x", object())
        expert = VLMVerificationExpert(backend, image_store=store_img)

        # Build a real scoped view via the reducer to satisfy validation.
        from rs_hmcomm.contracts import AnswerCandidate as AC, Claim as C, Evidence as E
        from rs_hmcomm.workspace import WorkspaceDelta, WorkspaceReducer

        task = TaskContext(
            task_id="t",
            image_ref="img://x",
            question="q",
            task_type="object_quantity",
            answer_format="short_text",
        )
        state = WorkspaceState.initial("ep-v", task)
        evidence = E(
            evidence_id="ev-1",
            kind="visual",
            summary="s",
            created_by="visual",
            created_step=0,
        )
        claim_real = C(
            claim_id="claim-1",
            statement="s",
            created_by="reasoning",
            evidence_ids=("ev-1",),
            created_step=1,
        )
        candidate = AC(
            candidate_id="reasoning-answer-s1",
            answer="5",
            created_by="reasoning",
            claim_ids=("claim-1",),
            created_step=1,
        )
        state = WorkspaceReducer.apply(
            state,
            WorkspaceDelta(
                delta_id="d1",
                episode_id="ep-v",
                base_version=0,
                actor_id="reasoning",
                evidence=(evidence,),
                claims=(claim_real,),
                answer_candidates=(candidate,),
            ),
        )
        store = ImageStore()
        store.put("img://x", object())
        observation = ExpertObservation(
            episode_id="ep-v",
            step=2,
            expert_id="verification",
            objective="verify",
            workspace=build_scoped_view(state, {"ev-1", "claim-1", "reasoning-answer-s1"}),
        )
        result = expert.run(observation)
        verified = result.workspace_delta.answer_candidates[0]
        assert verified.status.value == "verified"
        assert verified.answer == "2"

    def test_accept_keeps_source_answer(self):
        from rs_hmcomm.contracts import AnswerCandidate as AC, Claim as C, Evidence as E
        from rs_hmcomm.workspace import WorkspaceDelta, WorkspaceReducer

        backend = FakeBackend(['{"verdict": "accept"}'])
        store_img = ImageStore()
        store_img.put("img://x", object())
        expert = VLMVerificationExpert(backend, image_store=store_img)

        task = TaskContext(
            task_id="t",
            image_ref="img://x",
            question="q",
            task_type="object_existence",
            answer_format="short_text",
        )
        state = WorkspaceState.initial("ep-w", task)
        evidence = E(
            evidence_id="ev-1", kind="visual", summary="s",
            created_by="visual", created_step=0,
        )
        claim_real = C(
            claim_id="claim-1", statement="s", created_by="reasoning",
            evidence_ids=("ev-1",), created_step=1,
        )
        candidate = AC(
            candidate_id="reasoning-answer-s1", answer="Yes",
            created_by="reasoning", claim_ids=("claim-1",), created_step=1,
        )
        state = WorkspaceReducer.apply(
            state,
            WorkspaceDelta(
                delta_id="d1", episode_id="ep-w", base_version=0,
                actor_id="reasoning",
                evidence=(evidence,), claims=(claim_real,),
                answer_candidates=(candidate,),
            ),
        )
        store = ImageStore()
        store.put("img://x", object())
        observation = ExpertObservation(
            episode_id="ep-w", step=2, expert_id="verification",
            objective="v",
            workspace=build_scoped_view(state, {"ev-1", "claim-1", "reasoning-answer-s1"}),
        )
        result = expert.run(observation)
        verified = result.workspace_delta.answer_candidates[0]
        assert verified.answer == "Yes"
