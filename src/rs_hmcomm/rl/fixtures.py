"""Deterministic Phase 5 mechanism gate.

These experts do not claim visual accuracy.  They create controlled environment
dynamics in which quantity tasks should skip Knowledge while airport
classification tasks require it.  This makes routing learning measurable
without leaking the environment's reference answer into agent observations.
"""
from __future__ import annotations

from dataclasses import replace

from rs_hmcomm.artifacts import ArtifactKind, VisualArtifact
from rs_hmcomm.contracts import (
    AnswerCandidate,
    BudgetLimits,
    Claim,
    Evidence,
    RecordStatus,
    TaskContext,
)
from rs_hmcomm.control import EvolutionController, LearnedRoutingPolicy
from rs_hmcomm.experts import (
    ExpertObservation,
    ExpertResult,
    KnowledgeExpert,
    VerificationExpert,
    VisualExpert,
)
from rs_hmcomm.runtime import LangGraphMultiAgentRuntime, MultiAgentRunResult
from rs_hmcomm.workspace import WorkspaceDelta

from .routing import (
    RoutingTrainingSample,
    fingerprint_expert_types,
)


class Phase5FixtureVisualExpert(VisualExpert):
    """Add a descriptor-only latent so modality learning is causally testable.

    This remains a model-free mechanism fixture: no binary feature is fabricated.
    The large declared byte size makes Hybrid disclosure carry a measurable cost;
    a separate real-Qwen smoke proves that the same route loads physical latents.
    """

    def run(self, observation: ExpertObservation) -> ExpertResult:
        base = super().run(observation)
        artifact_id = f"phase5-latent-s{observation.step}"
        latent = VisualArtifact(
            artifact_id=artifact_id,
            kind=ArtifactKind.LATENT,
            media_type="application/x-evors-fixture-latent",
            shape=(62_500,),
            dtype="float32",
            byte_size=250_000,
            source_image_ref=observation.workspace.task.image_ref,
            roi=(0.0, 0.0, 1.0, 1.0),
            encoder="deterministic_phase5_descriptor",
            layer="fixture_only",
            projector="none",
            compatible_consumers=("reasoning",),
            sha256="f" * 64,
        )
        delta = replace(
            base.workspace_delta,
            evidence=tuple(
                replace(item, artifact_refs=(artifact_id,))
                for item in base.workspace_delta.evidence
            ),
            spatial_nodes=tuple(
                replace(item, visual_artifact_refs=(artifact_id,))
                for item in base.workspace_delta.spatial_nodes
            ),
            artifacts=(latent,),
        )
        result = ExpertResult(
            expert_id=self.expert_id,
            workspace_delta=delta,
            cost_delta=base.cost_delta,
            metrics={
                **base.metrics,
                "latent_fixture": "descriptor_only",
                "scientific_claim": False,
            },
        )
        result.validate_against(observation)
        return result


class Phase5FixtureReasoningExpert:
    """Reason over fixture evidence with a knowledge-dependent branch."""

    expert_id = "reasoning"

    def run(self, observation: ExpertObservation) -> ExpertResult:
        if not isinstance(observation, ExpertObservation):
            raise TypeError("observation must be an ExpertObservation")
        if observation.expert_id != self.expert_id:
            raise ValueError("fixture reasoning expert received the wrong observation")
        workspace = observation.workspace
        if not workspace.evidence:
            raise ValueError("fixture reasoning requires disclosed evidence")
        task_type = workspace.task.task_type.casefold()
        has_knowledge = any(item.created_by == "knowledge" for item in workspace.evidence)
        has_latent = any(item.kind == ArtifactKind.LATENT for item in workspace.artifacts)
        if "quantity" in task_type:
            answer = "6"
            statement = "Fixture visual evidence deterministically contains six objects."
        elif "classification" in task_type:
            answer = "civil_airport" if has_knowledge else "military_airport"
            statement = (
                "Fixture airport structure plus disclosed domain knowledge supports the class."
                if has_knowledge
                else "Fixture visual structure alone leaves the airport class ambiguous."
            )
        elif "existence" in task_type:
            # Keep the environment-only target absent from TaskContext.  Using
            # ``yes`` would collide with the public ``yes_no`` format string
            # and invalidate the label-isolation guard.
            answer = "present" if has_latent else "absent"
            statement = (
                "Authorized fixture latent plus visual structure supports existence."
                if has_latent
                else "Text and structure alone are insufficient for this fixture branch."
            )
        else:
            raise ValueError(f"unsupported Phase 5 fixture task type: {task_type}")
        evidence_ids = tuple(item.evidence_id for item in workspace.evidence)
        claim = Claim(
            claim_id=f"phase5-reasoning-claim-s{observation.step}",
            statement=statement,
            created_by=self.expert_id,
            evidence_ids=evidence_ids,
            created_step=observation.step,
            status=RecordStatus.PROPOSED,
        )
        candidate = AnswerCandidate(
            candidate_id=f"phase5-reasoning-answer-s{observation.step}",
            answer=answer,
            created_by=self.expert_id,
            claim_ids=(claim.claim_id,),
            evidence_ids=evidence_ids,
            created_step=observation.step,
            status=RecordStatus.PROPOSED,
        )
        delta = WorkspaceDelta(
            delta_id=f"phase5-reasoning-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            claims=(claim,),
            answer_candidates=(candidate,),
        )
        result = ExpertResult(expert_id=self.expert_id, workspace_delta=delta)
        result.validate_against(observation)
        return result


def phase5_fixture_experts() -> dict[str, object]:
    return {
        "visual": Phase5FixtureVisualExpert(),
        "knowledge": KnowledgeExpert(),
        "reasoning": Phase5FixtureReasoningExpert(),
        "verification": VerificationExpert(),
    }


class Phase5FixtureRunner:
    """Run the real LangGraph/Workspace/Router loop with frozen fixtures."""

    def __init__(self, *, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self._fingerprint = fingerprint_expert_types(phase5_fixture_experts())
        self.budget = BudgetLimits(
            max_agent_calls=max_steps,
            max_message_text_tokens=100_000,
            max_message_structure_bytes=100_000,
            max_message_latent_bytes=10_000_000,
            max_errors=1,
            max_invalid_actions=0,
        )

    def expert_fingerprint(self) -> dict[str, str]:
        return dict(self._fingerprint)

    def run(
        self,
        policy: LearnedRoutingPolicy,
        sample: RoutingTrainingSample,
        *,
        episode_id: str,
    ) -> MultiAgentRunResult:
        runtime = LangGraphMultiAgentRuntime(
            EvolutionController(policy),
            phase5_fixture_experts(),
            self.budget,
            max_steps=self.max_steps,
            disclosure_mode="scoped",
        )
        return runtime.run(sample.task, episode_id=episode_id)


def _sample(
    sample_id: str,
    *,
    question: str,
    task_type: str,
    reference: str,
) -> RoutingTrainingSample:
    task = TaskContext(
        task_id=f"phase5-task-{sample_id}",
        image_ref=f"fixture://phase5/{sample_id}",
        question=question,
        task_type=task_type,
        answer_format=(
            "integer"
            if "quantity" in task_type
            else "yes_no"
            if "existence" in task_type
            else "short_text"
        ),
        metadata={
            "mechanism_gate": True,
            "scientific_claim": False,
            "reference_available_to_agents": False,
        },
    )
    return RoutingTrainingSample(sample_id, task, reference)


def build_phase5_fixture_splits() -> tuple[
    tuple[RoutingTrainingSample, ...],
    tuple[RoutingTrainingSample, ...],
]:
    count_train = (
        "How many aircraft are visible?",
        "Count the aircraft in the remote-sensing scene.",
        "What is the number of visible aircraft?",
        "统计图像中的飞机数量。",
        "How many objects are present on the apron?",
        "Count all question-relevant objects.",
    )
    class_train = (
        "What type of airport is shown?",
        "Classify the airport category from its layout.",
        "判断该机场的类型。",
        "Determine the airport category.",
        "What airport class best matches the scene?",
        "Use the runway layout to classify this airport.",
    )
    existence_train = (
        "Is a question-relevant object visible in the scene?",
        "Does the image contain the requested object?",
        "Can an object be confirmed from the visual evidence?",
        "图中是否存在目标物体？",
        "Is the requested feature visible?",
        "Determine whether an object is visible in this image.",
    )
    count_eval = (
        "How many planes can be counted in this scene?",
        "请统计停机坪附近目标的数量。",
    )
    class_eval = (
        "Identify the type of this airport.",
        "根据跑道与滑行道判断机场类别。",
    )
    existence_eval = (
        "Is there a target object in this remote-sensing scene?",
        "图像中有没有指定目标？",
    )
    train = tuple(
        _sample(
            f"train-count-{index}",
            question=question,
            task_type="object_quantity",
            reference="6",
        )
        for index, question in enumerate(count_train)
    ) + tuple(
        _sample(
            f"train-class-{index}",
            question=question,
            task_type="airport_classification",
            reference="civil_airport",
        )
        for index, question in enumerate(class_train)
    ) + tuple(
        _sample(
            f"train-existence-{index}",
            question=question,
            task_type="object_existence",
            reference="present",
        )
        for index, question in enumerate(existence_train)
    )
    evaluate = tuple(
        _sample(
            f"eval-count-{index}",
            question=question,
            task_type="object_quantity",
            reference="6",
        )
        for index, question in enumerate(count_eval)
    ) + tuple(
        _sample(
            f"eval-class-{index}",
            question=question,
            task_type="airport_classification",
            reference="civil_airport",
        )
        for index, question in enumerate(class_eval)
    ) + tuple(
        _sample(
            f"eval-existence-{index}",
            question=question,
            task_type="object_existence",
            reference="present",
        )
        for index, question in enumerate(existence_eval)
    )
    return train, evaluate


__all__ = [
    "Phase5FixtureReasoningExpert",
    "Phase5FixtureVisualExpert",
    "Phase5FixtureRunner",
    "build_phase5_fixture_splits",
    "phase5_fixture_experts",
]
