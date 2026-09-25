"""Deterministic, model-free Cognitive Expert mechanisms for Phase 4 smoke."""
from __future__ import annotations

from rs_hmcomm.contracts import (
    AnswerCandidate,
    Claim,
    Evidence,
    RecordStatus,
)
from rs_hmcomm.spatial import SpatialLevel, SpatialNode
from rs_hmcomm.workspace import WorkspaceDelta

from .contracts import ExpertObservation, ExpertResult


class _ExpertMechanism:
    expert_id: str

    def _require_observation(self, observation: ExpertObservation) -> None:
        if not isinstance(observation, ExpertObservation):
            raise TypeError("observation must be an ExpertObservation")
        if observation.expert_id != self.expert_id:
            raise ValueError(
                f"{self.expert_id} expert cannot execute an "
                f"{observation.expert_id} observation"
            )

    def _result(
        self,
        observation: ExpertObservation,
        delta: WorkspaceDelta,
    ) -> ExpertResult:
        result = ExpertResult(expert_id=self.expert_id, workspace_delta=delta)
        result.validate_against(observation)
        return result


class VisualExpert(_ExpertMechanism):
    """Create a minimal Scene/Object description and visual Evidence.

    This Phase 4 mechanism does not inspect pixels.  Its evidence is explicitly
    marked as a deterministic integration fixture, not a scientific visual
    model result.
    """

    expert_id = "visual"

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        evidence_id = f"visual-evidence-s{observation.step}"
        object_id = f"visual-object-s{observation.step}"

        existing_scenes = tuple(
            item for item in workspace.spatial_nodes if item.level == SpatialLevel.SCENE
        )
        if existing_scenes:
            scene_id = existing_scenes[0].node_id
            scene_additions: tuple[SpatialNode, ...] = ()
        else:
            scene_id = f"visual-scene-s{observation.step}"
            scene_additions = (
                SpatialNode(
                    node_id=scene_id,
                    level=SpatialLevel.SCENE,
                    semantic="question-conditioned remote-sensing scene",
                    provenance={
                        "mechanism": "deterministic_rule_fixture",
                        "scientific_claim": False,
                    },
                ),
            )

        evidence = Evidence(
            evidence_id=evidence_id,
            kind="visual",
            summary=(
                "Deterministic visual integration fixture for objective: "
                f"{observation.objective}"
            ),
            created_by=self.expert_id,
            created_step=observation.step,
            spatial_node_ids=(object_id,),
            payload={
                "mechanism": "deterministic_rule_fixture",
                "pixel_model_used": False,
                "scientific_claim": False,
            },
        )
        object_node = SpatialNode(
            node_id=object_id,
            level=SpatialLevel.OBJECT,
            semantic="question-relevant visual object",
            bbox=(0.0, 0.0, 1.0, 1.0),
            parent_id=scene_id,
            evidence_ids=(evidence_id,),
            provenance={
                "mechanism": "deterministic_rule_fixture",
                "scientific_claim": False,
            },
        )
        delta = WorkspaceDelta(
            delta_id=f"visual-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            evidence=(evidence,),
            spatial_nodes=(*scene_additions, object_node),
        )
        return self._result(observation, delta)


class KnowledgeExpert(_ExpertMechanism):
    """Add a small built-in remote-sensing domain-knowledge record."""

    expert_id = "knowledge"

    @staticmethod
    def _knowledge_for(question: str) -> str:
        lowered = question.casefold()
        if any(term in lowered for term in ("airport", "runway", "机场", "跑道")):
            return (
                "Airport interpretation commonly considers runways, taxiways, "
                "aprons, and their spatial arrangement."
            )
        if any(term in lowered for term in ("aircraft", "airplane", "飞机")):
            return (
                "Aircraft observations should be distinguished from nearby "
                "vehicles and should not be double-counted across image crops."
            )
        return (
            "Remote-sensing interpretation should combine visible evidence "
            "with task-relevant spatial context."
        )

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        evidence = Evidence(
            evidence_id=f"knowledge-evidence-s{observation.step}",
            kind="domain_knowledge",
            summary=self._knowledge_for(workspace.task.question),
            created_by=self.expert_id,
            created_step=observation.step,
            payload={
                "source": "builtin_phase4_rule",
                "model_used": False,
            },
        )
        delta = WorkspaceDelta(
            delta_id=f"knowledge-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            evidence=(evidence,),
        )
        return self._result(observation, delta)


class ReasoningExpert(_ExpertMechanism):
    """Form a Claim and AnswerCandidate from explicitly visible Evidence."""

    expert_id = "reasoning"

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        if not workspace.evidence:
            raise ValueError("reasoning expert requires at least one visible evidence record")

        metadata = workspace.task.metadata
        scripted = metadata.get("scripted_fixture") is True
        mock_answer = metadata.get("mock_answer")
        if mock_answer is not None and not scripted:
            raise ValueError(
                "task.metadata.mock_answer may only be used by a scripted_fixture"
            )
        if scripted:
            if mock_answer is None or not str(mock_answer).strip():
                raise ValueError("scripted_fixture requires a non-empty mock_answer")
            answer = str(mock_answer)
            statement = (
                "SCRIPTED FIXTURE ONLY (non-scientific): visible evidence is "
                "paired with the fixture answer for control-flow validation."
            )
        else:
            # A model-free rule mechanism cannot honestly infer an open-ended
            # VQA answer.  It still exercises Claim/Answer plumbing without
            # reading evaluation labels or pretending to be a scientific run.
            answer = "requires_model_reasoning"
            summaries = "; ".join(item.summary for item in workspace.evidence)
            statement = f"Visible evidence considered by the rule mechanism: {summaries}"

        evidence_ids = tuple(item.evidence_id for item in workspace.evidence)
        claim = Claim(
            claim_id=f"reasoning-claim-s{observation.step}",
            statement=statement,
            created_by=self.expert_id,
            evidence_ids=evidence_ids,
            created_step=observation.step,
            status=RecordStatus.PROPOSED,
        )
        candidate = AnswerCandidate(
            candidate_id=f"reasoning-answer-s{observation.step}",
            answer=answer,
            created_by=self.expert_id,
            claim_ids=(claim.claim_id,),
            evidence_ids=evidence_ids,
            created_step=observation.step,
            status=RecordStatus.PROPOSED,
        )
        delta = WorkspaceDelta(
            delta_id=f"reasoning-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            claims=(claim,),
            answer_candidates=(candidate,),
        )
        return self._result(observation, delta)


class VerificationExpert(_ExpertMechanism):
    """Promote one visible candidate to a new VERIFIED candidate record."""

    expert_id = "verification"

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        if not workspace.answer_candidates:
            raise ValueError("verification expert requires a visible answer candidate")

        source = sorted(
            workspace.answer_candidates,
            key=lambda item: item.candidate_id,
        )[-1]
        verified = AnswerCandidate(
            candidate_id=f"{source.candidate_id}-verified-s{observation.step}",
            answer=source.answer,
            created_by=self.expert_id,
            claim_ids=source.claim_ids,
            evidence_ids=source.evidence_ids,
            created_step=observation.step,
            confidence=source.confidence,
            status=RecordStatus.VERIFIED,
        )
        delta = WorkspaceDelta(
            delta_id=f"verification-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            answer_candidates=(verified,),
        )
        return self._result(observation, delta)


__all__ = [
    "KnowledgeExpert",
    "ReasoningExpert",
    "VerificationExpert",
    "VisualExpert",
]
