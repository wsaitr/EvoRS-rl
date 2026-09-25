"""Adapter exposing the completed single-Agent evolution loop as Visual Expert.

The inner Generalist performs active perception and private evolutionary
selection. Only its selected evidence/representation records are committed to
the multi-Agent Workspace; its answer candidate remains private and the
Reasoning Expert retains ownership of the public answer candidate.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from rs_hmcomm.artifacts import ArtifactKind, InMemoryArtifactStore
from rs_hmcomm.contracts import CostVector, Evidence
from rs_hmcomm.workspace import WorkspaceDelta

from .contracts import ExpertObservation, ExpertResult
from .image_store import ImageStore


class EvolvingVisualExpert:
    """Visual Expert backed by real active perception and intra-Agent evolution."""

    expert_id = "visual"

    def __init__(
        self,
        runtime: Any,
        *,
        image_store: ImageStore,
        artifact_store: InMemoryArtifactStore,
    ) -> None:
        if not callable(getattr(runtime, "run", None)):
            raise TypeError("runtime must implement run")
        if not isinstance(image_store, ImageStore):
            raise TypeError("image_store must be an ImageStore")
        if not isinstance(artifact_store, InMemoryArtifactStore):
            raise TypeError("artifact_store must be an InMemoryArtifactStore")
        self.runtime = runtime
        self.image_store = image_store
        self.artifact_store = artifact_store

    def run(self, observation: ExpertObservation) -> ExpertResult:
        if not isinstance(observation, ExpertObservation):
            raise TypeError("observation must be an ExpertObservation")
        if observation.expert_id != self.expert_id:
            raise ValueError("EvolvingVisualExpert requires a visual observation")
        task = observation.workspace.task
        image = self.image_store.get(task.image_ref)
        if image is None:
            raise ValueError(f"image_ref is not registered: {task.image_ref}")

        inner = self.runtime.run(
            image,
            task.question,
            sample_id=f"{task.task_id}-visual-s{observation.step}",
            image_ref=task.image_ref,
            task_type=task.task_type,
            answer_format=task.answer_format,
            task_metadata={"outer_episode_id": observation.episode_id},
        )

        # Move physical artifacts into the episode-level store. Workspace only
        # receives the immutable descriptors below.
        for artifact in inner.representation_view.artifacts:
            stored = inner.artifact_store.get(artifact.artifact_id)
            self.artifact_store.put(stored.artifact, stored.data)

        evidence = tuple(
            Evidence(
                evidence_id=item.evidence_id,
                kind=item.kind,
                summary=item.summary,
                created_by=self.expert_id,
                created_step=observation.step,
                confidence=item.confidence,
                spatial_node_ids=item.spatial_node_ids,
                artifact_refs=item.artifact_refs,
                payload={
                    **item.payload,
                    "inner_agent": "integrated-generalist",
                    "private_evolution_committed": False,
                },
            )
            for item in inner.representation_view.evidence
        )
        nodes = tuple(
            replace(
                item,
                provenance={
                    **item.provenance,
                    "created_by": self.expert_id,
                    "inner_agent": "integrated-generalist",
                },
            )
            for item in inner.representation_view.spatial_nodes
        )
        delta = WorkspaceDelta(
            delta_id=f"visual-evolution-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=observation.workspace.version,
            actor_id=self.expert_id,
            evidence=evidence,
            spatial_nodes=nodes,
            artifacts=inner.representation_view.artifacts,
        )
        evolution = inner.evolution
        metrics = {
            "mechanism": "active_perception_with_private_evolution_v1",
            "decision_rule": inner.perception.decision_rule,
            "perception_answer": inner.perception.answer,
            "selected_private_answer": inner.answer,
            "model_calls": inner.perception.model_calls,
            "tool_calls": inner.perception.tool_calls,
            "candidate_count": (
                evolution.metrics.candidates_evaluated if evolution is not None else 1
            ),
            "variation_count": (
                evolution.metrics.variations_applied if evolution is not None else 0
            ),
            "generation_count": (
                evolution.metrics.generations_completed if evolution is not None else 0
            ),
            "latent_enabled": inner.latent.enabled,
            "latent_bytes_loaded": inner.latent.bytes_loaded,
            "workspace_answer_candidate_committed": False,
        }
        inner_cost = inner.cost
        result = ExpertResult(
            expert_id=self.expert_id,
            workspace_delta=delta,
            cost_delta=CostVector(
                tool_calls=inner_cost.tool_calls,
                model_calls=inner_cost.model_calls,
                model_input_tokens=inner_cost.model_input_tokens,
                model_output_tokens=inner_cost.model_output_tokens,
                billed_compute_ms=inner_cost.billed_compute_ms,
            ),
            metrics=metrics,
        )
        result.validate_against(observation)
        if any(
            artifact.kind == ArtifactKind.LATENT
            and "reasoning" not in artifact.compatible_consumers
            for artifact in delta.artifacts
        ):
            raise ValueError("Visual Expert latent must authorize the reasoning consumer")
        return result


__all__ = ["EvolvingVisualExpert"]
