"""Integrated Phase 1-3 runtime for one EvoRS Generalist Agent.

The runtime keeps the paper's two boundaries explicit:

* private candidate evolution stays inside one Agent activation;
* only selected records cross into the episode-scoped Workspace.

It deliberately contains no Controller, Router, inbox, or peer Agent calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import io
import json
import math
import re
import time
from typing import Any, Callable, Mapping

from PIL import Image

from rs_hmcomm.agents.active_perception import (
    ActivePerceptionAgent,
    ActivePerceptionResult,
)
from rs_hmcomm.artifacts import (
    ArtifactKind,
    InMemoryArtifactStore,
    LatentConsumer,
    LatentEncoder,
    LatentFeatureConfig,
    VisualArtifact,
)
from rs_hmcomm.contracts import (
    AnswerCandidate,
    Claim,
    CostVector,
    Evidence,
    RecordStatus,
    TaskContext,
)
from rs_hmcomm.evolution import (
    Candidate,
    EvolutionBudget,
    EvolutionConfig,
    EvolutionRunResult,
    FitnessEvaluation,
    IntraAgentEvolution,
    VariationContext,
    VariationOutcome,
)
from rs_hmcomm.spatial import SpatialLevel, SpatialNode
from rs_hmcomm.workspace import (
    ScopedWorkspaceView,
    WorkspaceDelta,
    WorkspaceReducer,
    WorkspaceState,
    build_scoped_view,
)


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TARGET_RE = re.compile(
    r"(?:how many|number of)\s+(.+?)(?:\s+(?:are|is|can|could|do|does)\b|\?|$)",
    flags=re.IGNORECASE,
)


def _safe_id(value: str) -> str:
    normalized = _SAFE_ID_RE.sub("-", value.strip()).strip("-")
    return normalized or "sample"


def _target_semantic(question: str) -> str:
    match = _TARGET_RE.search(question)
    if match is None:
        return "question_relevant_object"
    return match.group(1).strip().lower().replace(" ", "_")


def _image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _put_image_artifact(
    store: InMemoryArtifactStore,
    image: Image.Image,
    *,
    artifact_id: str,
    kind: ArtifactKind,
    source_image_ref: str,
    roi: tuple[float, float, float, float] | None,
) -> VisualArtifact:
    rgb = image.convert("RGB")
    data = _image_bytes(rgb)
    descriptor = VisualArtifact(
        artifact_id=artifact_id,
        kind=kind,
        media_type="image/png",
        shape=(rgb.height, rgb.width, 3),
        dtype="uint8",
        byte_size=len(data),
        source_image_ref=source_image_ref,
        roi=roi,
        encoder=None,
        layer=None,
        projector=None,
        compatible_consumers=(),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return store.put(descriptor, data)


def _crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    return image.crop(
        (
            math.floor(bbox[0] * width),
            math.floor(bbox[1] * height),
            math.ceil(bbox[2] * width),
            math.ceil(bbox[3] * height),
        )
    )


@dataclass(frozen=True, slots=True)
class IntegratedSingleAgentConfig:
    """Feature flags and bounded private-search settings."""

    evolution_enabled: bool = True
    max_candidate_variations: int = 4
    latent_enabled: bool = False
    latent_consumer_id: str = "integrated-generalist"
    max_spatial_objects: int = 8
    create_group_for_repeated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.evolution_enabled, bool):
            raise TypeError("evolution_enabled must be a boolean")
        if (
            isinstance(self.max_candidate_variations, bool)
            or not isinstance(self.max_candidate_variations, int)
            or self.max_candidate_variations < 1
        ):
            raise ValueError("max_candidate_variations must be a positive integer")
        if not isinstance(self.latent_enabled, bool):
            raise TypeError("latent_enabled must be a boolean")
        if (
            not isinstance(self.latent_consumer_id, str)
            or not self.latent_consumer_id.strip()
            or any(character.isspace() for character in self.latent_consumer_id)
        ):
            raise ValueError("latent_consumer_id must be a non-empty ID")
        if (
            isinstance(self.max_spatial_objects, bool)
            or not isinstance(self.max_spatial_objects, int)
            or self.max_spatial_objects < 1
        ):
            raise ValueError("max_spatial_objects must be a positive integer")
        if not isinstance(self.create_group_for_repeated, bool):
            raise TypeError("create_group_for_repeated must be a boolean")


@dataclass(frozen=True, slots=True)
class LatentDiagnostics:
    enabled: bool
    learned_representation: bool
    artifact_ids: tuple[str, ...] = ()
    bytes_loaded: int = 0
    value_count: int = 0
    roi_latent_count: int = 0
    mean_roi_cosine_similarity: float | None = None
    value_sha256: str | None = None
    influenced_fitness: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "learned_representation": self.learned_representation,
            "artifact_ids": list(self.artifact_ids),
            "bytes_loaded": self.bytes_loaded,
            "value_count": self.value_count,
            "roi_latent_count": self.roi_latent_count,
            "mean_roi_cosine_similarity": self.mean_roi_cosine_similarity,
            "value_sha256": self.value_sha256,
            "influenced_fitness": self.influenced_fitness,
        }


@dataclass(frozen=True, slots=True)
class IntegratedSingleAgentResult:
    task: TaskContext
    answer: str
    perception: ActivePerceptionResult
    evolution: EvolutionRunResult | None
    workspace: WorkspaceState
    scoped_view: ScopedWorkspaceView
    representation_view: ScopedWorkspaceView
    cost: CostVector
    latent: LatentDiagnostics
    wall_time_ms: float
    artifact_store: InMemoryArtifactStore = field(repr=False, compare=False)

    def to_dict(self, *, include_private_evolution: bool = False) -> dict[str, Any]:
        perception = {
            "answer": self.perception.answer,
            "decision_rule": self.perception.decision_rule,
            "tool_calls": self.perception.tool_calls,
            "model_calls": self.perception.model_calls,
            "candidate_boxes": [list(box) for box in self.perception.candidate_boxes],
            "observations": [item.to_dict() for item in self.perception.calls],
        }
        evolution: dict[str, Any] | None = None
        if self.evolution is not None:
            evolution = self.evolution.to_dict()
            if include_private_evolution:
                evolution["private_trace"] = self.evolution.private_trace.to_dict()
        workspace_dict = self.workspace.to_dict()
        return {
            "schema_version": "evors.integrated-single-agent.v1",
            "task": self.task.to_dict(),
            "answer": self.answer,
            "perception": perception,
            "evolution": evolution,
            "workspace": workspace_dict,
            "workspace_contains_binary_payload": any(
                marker in json.dumps(workspace_dict, ensure_ascii=False)
                for marker in ("artifact_payload", "pixel_values", "tensor_values")
            ),
            "representation_view": self.representation_view.to_dict(),
            "scoped_view": self.scoped_view.to_dict(),
            "cost": self.cost.to_dict(),
            "latent": self.latent.to_dict(),
            "artifact_store_size": len(self.artifact_store),
            "wall_time_ms": self.wall_time_ms,
        }


class _EvidenceGuidedVariation:
    """Replace the answer genome with another observed, evidence-backed answer."""

    name = "evidence_guided_answer_revision"

    def __init__(
        self,
        alternatives: tuple[str, ...],
        sources_by_answer: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.alternatives = alternatives
        self.sources_by_answer = dict(sources_by_answer)

    def vary(
        self,
        parent: Candidate,
        *,
        context: VariationContext,
    ) -> VariationOutcome:
        start = (context.generation - 1) % len(self.alternatives)
        answer = self.alternatives[start]
        if answer == parent.payload.get("answer") and len(self.alternatives) > 1:
            answer = self.alternatives[(start + 1) % len(self.alternatives)]
        return VariationOutcome(
            payload={
                "answer": answer,
                "evidence_sources": list(self.sources_by_answer.get(answer, ())),
            }
        )


class _PerceptionFitness:
    """Label-free fitness over perceptual, structural, latent, and cost support."""

    name = "perception_structure_latent_cost_fitness_v1"

    def __init__(
        self,
        *,
        support_by_answer: Mapping[str, float],
        active_answer: str,
        spatial_object_count: int,
        roi_latent_count: int,
        source_cost_by_answer: Mapping[str, float],
    ) -> None:
        self.support_by_answer = dict(support_by_answer)
        self.active_answer = active_answer
        self.spatial_object_count = spatial_object_count
        self.roi_latent_count = roi_latent_count
        self.source_cost_by_answer = dict(source_cost_by_answer)

    @staticmethod
    def _integer(answer: str) -> int | None:
        stripped = answer.strip()
        return int(stripped) if stripped.isdigit() else None

    def evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        answer = str(candidate.payload.get("answer", "")).strip()
        integer = self._integer(answer)
        evidence_support = self.support_by_answer.get(answer, 0.0)
        active_decision = float(answer == self.active_answer)
        structure_consistency = float(
            integer is not None
            and self.spatial_object_count > 0
            and integer == self.spatial_object_count
        )
        latent_consistency = float(
            integer is not None
            and self.roi_latent_count > 0
            and integer == self.roi_latent_count
        )
        cost_efficiency = 1.0 / (
            1.0 + self.source_cost_by_answer.get(answer, 1.0)
        )
        answer_format = float(bool(answer))
        score = (
            evidence_support
            + 4.0 * active_decision
            + 0.75 * structure_consistency
            + 0.25 * latent_consistency
            + 0.10 * cost_efficiency
            + 0.10 * answer_format
        )
        return FitnessEvaluation(
            score=score,
            components={
                "evidence_support": evidence_support,
                "active_decision": active_decision,
                "structure_consistency": structure_consistency,
                "latent_consistency": latent_consistency,
                "cost_efficiency": cost_efficiency,
                "answer_format": answer_format,
            },
        )


class IntegratedSingleAgentRuntime:
    """Run one active-perception Generalist through the complete Phase 1-3 loop."""

    agent_id = "integrated-generalist"

    def __init__(
        self,
        perception_agent: ActivePerceptionAgent,
        *,
        config: IntegratedSingleAgentConfig | None = None,
        latent_encoder_factory: (
            Callable[[InMemoryArtifactStore], LatentEncoder] | None
        ) = None,
    ) -> None:
        if not isinstance(perception_agent, ActivePerceptionAgent):
            raise TypeError("perception_agent must be an ActivePerceptionAgent")
        self.perception_agent = perception_agent
        self.config = config or IntegratedSingleAgentConfig()
        self.latent_encoder_factory = latent_encoder_factory
        if self.config.latent_enabled and latent_encoder_factory is None:
            raise ValueError("latent_enabled requires an explicit latent_encoder_factory")

    @staticmethod
    def _add_support(
        support: dict[str, float],
        sources: dict[str, list[str]],
        source_cost: dict[str, float],
        answer: str | None,
        *,
        weight: float,
        source: str,
        cost_rank: float,
    ) -> None:
        if answer is None or not answer.strip():
            return
        answer = answer.strip()
        support[answer] = support.get(answer, 0.0) + weight
        sources.setdefault(answer, []).append(source)
        source_cost[answer] = min(source_cost.get(answer, cost_rank), cost_rank)

    def _candidate_evidence(
        self,
        perception: ActivePerceptionResult,
    ) -> tuple[dict[str, float], dict[str, tuple[str, ...]], dict[str, float]]:
        support: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        source_cost: dict[str, float] = {}
        self._add_support(
            support,
            sources,
            source_cost,
            perception.answer,
            weight=3.0,
            source=f"selected:{perception.decision_rule}",
            cost_rank=float(perception.model_calls),
        )

        local_tile_counts: list[int] = []
        for call in perception.calls:
            answer: str | None = None
            weight = 1.0
            cost_rank = 1.0
            if call.view in {"object_localization", "enhanced_object_localization"}:
                boxes = ActivePerceptionAgent._localized_boxes(call.answer)
                answer = str(len(boxes)) if boxes is not None else None
                weight = 3.0 if call.view.startswith("enhanced") else 1.5
                cost_rank = 2.0
            elif call.view.startswith("tile_localization_"):
                continue
            elif call.view.startswith("tile_r"):
                count = ActivePerceptionAgent._count(call.answer)
                if count is not None:
                    local_tile_counts.append(count)
                continue
            else:
                count = ActivePerceptionAgent._count(call.answer)
                answer = str(count) if count is not None else None
                if call.view == "enhanced_global_2x":
                    weight, cost_rank = 2.0, 1.5
                elif call.view in {"candidate_contact_sheet", "visual_adjudication"}:
                    weight, cost_rank = 4.0, 3.0
            self._add_support(
                support,
                sources,
                source_cost,
                answer,
                weight=weight,
                source=call.view,
                cost_rank=cost_rank,
            )

        if local_tile_counts:
            self._add_support(
                support,
                sources,
                source_cost,
                str(sum(local_tile_counts)),
                weight=2.0,
                source="non_overlapping_tile_sum",
                cost_rank=3.0,
            )
        if perception.candidate_boxes:
            self._add_support(
                support,
                sources,
                source_cost,
                str(len(perception.candidate_boxes)),
                weight=1.5,
                source="deduplicated_spatial_candidates",
                cost_rank=2.5,
            )
        return (
            support,
            {answer: tuple(items) for answer, items in sources.items()},
            source_cost,
        )

    def _evolve(
        self,
        perception: ActivePerceptionResult,
        *,
        spatial_object_count: int,
        roi_latent_count: int,
    ) -> EvolutionRunResult | None:
        if not self.config.evolution_enabled:
            return None
        support, sources, source_cost = self._candidate_evidence(perception)
        alternatives = tuple(
            answer for answer in support if answer != perception.answer
        )[: self.config.max_candidate_variations]
        if not alternatives:
            return None
        generations = len(alternatives)
        seed = Candidate(
            candidate_id="private-perception-seed",
            payload={
                "answer": perception.answer,
                "evidence_sources": list(sources.get(perception.answer, ())),
            },
        )
        engine = IntraAgentEvolution(
            _EvidenceGuidedVariation(alternatives, sources),
            _PerceptionFitness(
                support_by_answer=support,
                active_answer=perception.answer,
                spatial_object_count=spatial_object_count,
                roi_latent_count=roi_latent_count,
                source_cost_by_answer=source_cost,
            ),
            EvolutionConfig(
                enabled=True,
                population_size=2,
                max_generations=generations,
                no_progress_generations=generations,
                min_fitness_gain=0.0,
                seed=0,
                budget=EvolutionBudget(
                    max_candidate_evaluations=generations + 1,
                    max_variations=generations,
                ),
            ),
        )
        return engine.run(seed)

    @staticmethod
    def _mean_cosine(vectors: list[tuple[float, ...]]) -> float | None:
        if len(vectors) < 2:
            return None
        values: list[float] = []
        for left_index, left in enumerate(vectors[:-1]):
            for right in vectors[left_index + 1 :]:
                dot = sum(a * b for a, b in zip(left, right))
                left_norm = math.sqrt(sum(value * value for value in left))
                right_norm = math.sqrt(sum(value * value for value in right))
                if left_norm > 0.0 and right_norm > 0.0:
                    values.append(dot / (left_norm * right_norm))
        return sum(values) / len(values) if values else None

    def _consume_latents(
        self,
        view: ScopedWorkspaceView,
        store: InMemoryArtifactStore,
    ) -> LatentDiagnostics:
        latent_artifacts = tuple(
            item for item in view.artifacts if item.kind == ArtifactKind.LATENT
        )
        if not self.config.latent_enabled:
            if latent_artifacts:
                raise RuntimeError("latent descriptors exist while latent is disabled")
            return LatentDiagnostics(enabled=False, learned_representation=False)
        consumer = LatentConsumer(
            store,
            self.config.latent_consumer_id,
            LatentFeatureConfig(enabled=True),
        )
        reads = [consumer.consume(item.artifact_id) for item in latent_artifacts]
        roi_vectors = [
            read.values
            for item, read in zip(latent_artifacts, reads)
            if item.roi is not None
        ]
        digest = hashlib.sha256(
            "".join(item.sha256 for item in latent_artifacts).encode("ascii")
        ).hexdigest()
        learned = bool(latent_artifacts) and all(
            item.encoder is not None and not item.encoder.startswith("pixel_stats")
            for item in latent_artifacts
        )
        return LatentDiagnostics(
            enabled=True,
            learned_representation=learned,
            artifact_ids=tuple(item.artifact_id for item in latent_artifacts),
            bytes_loaded=sum(item.bytes_loaded for item in reads),
            value_count=sum(len(item.values) for item in reads),
            roi_latent_count=len(roi_vectors),
            mean_roi_cosine_similarity=self._mean_cosine(roi_vectors),
            value_sha256=digest,
        )

    def run(
        self,
        image: Image.Image,
        question: str,
        *,
        sample_id: str = "sample-0",
        image_ref: str | None = None,
        task_type: str = "object_quantity",
        answer_format: str | None = None,
        task_metadata: Mapping[str, Any] | None = None,
    ) -> IntegratedSingleAgentResult:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        started = time.perf_counter()
        token = _safe_id(sample_id)
        task = TaskContext(
            task_id=f"task-{token}",
            image_ref=image_ref or f"runtime://{token}",
            question=question,
            task_type=task_type,
            answer_format=answer_format
            or ("integer" if task_type == "object_quantity" else "yes_no"),
            metadata=dict(task_metadata or {}),
        )
        episode_id = f"episode-{token}"
        state = WorkspaceState.initial(episode_id, task)
        store = InMemoryArtifactStore()
        latent_encoder = (
            self.latent_encoder_factory(store)
            if self.latent_encoder_factory is not None
            else None
        )

        perception = self.perception_agent.run(image, question, task_type)
        boxes = tuple(perception.candidate_boxes[: self.config.max_spatial_objects])
        final_evidence_id = f"evidence-perception-{token}"
        artifacts: list[VisualArtifact] = []
        scene_image = _put_image_artifact(
            store,
            image,
            artifact_id=f"image-scene-{token}",
            kind=ArtifactKind.IMAGE,
            source_image_ref=task.image_ref,
            roi=None,
        )
        artifacts.append(scene_image)

        scene_refs = [scene_image.artifact_id]
        if self.config.latent_enabled:
            assert latent_encoder is not None
            scene_latent = latent_encoder.encode(
                image,
                artifact_id=f"latent-scene-{token}",
                source_image_ref=task.image_ref,
                roi=None,
                compatible_consumers=(self.config.latent_consumer_id,),
            )
            artifacts.append(scene_latent)
            scene_refs.append(scene_latent.artifact_id)

        target = _target_semantic(question)
        scene = SpatialNode(
            node_id=f"scene-{token}",
            level=SpatialLevel.SCENE,
            semantic="remote_sensing_scene",
            evidence_ids=(final_evidence_id,),
            visual_artifact_refs=tuple(scene_refs),
            provenance={
                "question_conditioned": True,
                "decision_rule": perception.decision_rule,
            },
        )
        nodes: list[SpatialNode] = [scene]
        group: SpatialNode | None = None
        if len(boxes) > 1 and self.config.create_group_for_repeated:
            group = SpatialNode(
                node_id=f"group-{token}",
                level=SpatialLevel.GROUP,
                semantic=f"{target}_collection",
                parent_id=scene.node_id,
                evidence_ids=(final_evidence_id,),
                provenance={"member_count": len(boxes), "compression_role": True},
            )
            nodes.append(group)

        object_nodes: list[SpatialNode] = []
        for index, bbox in enumerate(boxes, start=1):
            crop_artifact = _put_image_artifact(
                store,
                _crop(image, bbox),
                artifact_id=f"crop-object-{token}-{index}",
                kind=ArtifactKind.CROP,
                source_image_ref=scene_image.artifact_id,
                roi=bbox,
            )
            artifacts.append(crop_artifact)
            object_refs = [crop_artifact.artifact_id]
            if self.config.latent_enabled:
                assert latent_encoder is not None
                latent = latent_encoder.encode(
                    image,
                    artifact_id=f"latent-object-{token}-{index}",
                    source_image_ref=task.image_ref,
                    roi=bbox,
                    compatible_consumers=(self.config.latent_consumer_id,),
                )
                artifacts.append(latent)
                object_refs.append(latent.artifact_id)
            node = SpatialNode(
                node_id=f"object-{token}-{index}",
                level=SpatialLevel.OBJECT,
                semantic=target,
                bbox=bbox,
                parent_id=group.node_id if group is not None else scene.node_id,
                evidence_ids=(final_evidence_id,),
                visual_artifact_refs=tuple(object_refs),
                provenance={
                    "candidate_index": index,
                    "source": "active_perception_localization",
                },
            )
            nodes.append(node)
            object_nodes.append(node)

        call_evidence = tuple(
            Evidence(
                evidence_id=f"evidence-view-{token}-{index}",
                kind="vlm_visual_observation",
                summary=f"{call.view}: {call.answer}",
                created_by=self.agent_id,
                created_step=index,
                payload={
                    "view": call.view,
                    "answer": call.answer,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "latency_ms": call.latency_ms,
                    "model_id": call.model_id,
                },
            )
            for index, call in enumerate(perception.calls, start=1)
        )
        final_evidence = Evidence(
            evidence_id=final_evidence_id,
            kind="active_perception_result",
            summary=(
                f"Active perception selected answer {perception.answer!r} via "
                f"{perception.decision_rule}."
            ),
            created_by=self.agent_id,
            created_step=len(perception.calls) + 1,
            spatial_node_ids=tuple(node.node_id for node in nodes),
            artifact_refs=tuple(item.artifact_id for item in artifacts),
            payload={
                "answer": perception.answer,
                "decision_rule": perception.decision_rule,
                "candidate_boxes": [list(box) for box in boxes],
                "model_calls": perception.model_calls,
                "tool_calls": perception.tool_calls,
            },
        )
        perception_evidence = call_evidence + (final_evidence,)
        state = WorkspaceReducer.apply(
            state,
            WorkspaceDelta(
                delta_id=f"delta-perception-{token}",
                episode_id=episode_id,
                base_version=state.version,
                actor_id=self.agent_id,
                evidence=perception_evidence,
                spatial_nodes=tuple(nodes),
                artifacts=tuple(artifacts),
            ),
        )
        representation_view = build_scoped_view(
            state,
            tuple(item.evidence_id for item in perception_evidence),
        )
        latent = self._consume_latents(representation_view, store)

        evolution = self._evolve(
            perception,
            spatial_object_count=len(object_nodes),
            roi_latent_count=latent.roi_latent_count,
        )
        selected_answer = (
            str(evolution.selected_candidate.payload["answer"])
            if evolution is not None
            else perception.answer
        )
        latent_influenced = False
        if evolution is not None and latent.roi_latent_count > 0:
            latent_components = {
                float(record.evaluation.components.get("latent_consistency", 0.0))
                for generation in evolution.private_trace.generations
                for record in generation.fitness_evaluations
            }
            latent_influenced = len(latent_components) > 1
        latent = replace(latent, influenced_fitness=latent_influenced)

        evidence_ids = tuple(item.evidence_id for item in perception_evidence)
        claim = Claim(
            claim_id=f"claim-answer-{token}",
            statement=(
                f"The selected answer is {selected_answer!r}, supported by active "
                "multi-view perception and the question-conditioned hierarchy."
            ),
            created_by=self.agent_id,
            evidence_ids=evidence_ids,
            created_step=len(perception.calls) + 2,
            status=RecordStatus.VERIFIED,
        )
        answer = AnswerCandidate(
            candidate_id=f"answer-{token}",
            answer=selected_answer,
            created_by=self.agent_id,
            claim_ids=(claim.claim_id,),
            evidence_ids=(final_evidence.evidence_id,),
            created_step=len(perception.calls) + 2,
            status=RecordStatus.VERIFIED,
        )
        state = WorkspaceReducer.apply(
            state,
            WorkspaceDelta(
                delta_id=f"delta-answer-{token}",
                episode_id=episode_id,
                base_version=state.version,
                actor_id=self.agent_id,
                claims=(claim,),
                answer_candidates=(answer,),
            ),
        )
        final_view = build_scoped_view(state, (answer.candidate_id,))
        wall_time_ms = (time.perf_counter() - started) * 1000.0
        cost = CostVector(
            agent_calls=1,
            tool_calls=perception.tool_calls,
            model_calls=perception.model_calls,
            model_input_tokens=perception.input_tokens,
            model_output_tokens=perception.output_tokens,
            billed_compute_ms=wall_time_ms,
        )
        return IntegratedSingleAgentResult(
            task=task,
            answer=selected_answer,
            perception=perception,
            evolution=evolution,
            workspace=state,
            scoped_view=final_view,
            representation_view=representation_view,
            cost=cost,
            latent=latent,
            wall_time_ms=wall_time_ms,
            artifact_store=store,
        )


__all__ = [
    "IntegratedSingleAgentConfig",
    "IntegratedSingleAgentResult",
    "IntegratedSingleAgentRuntime",
    "LatentDiagnostics",
]
