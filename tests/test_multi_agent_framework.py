from __future__ import annotations

from PIL import Image

from rs_hmcomm.agents import ActivePerceptionAgent, ActivePerceptionResult, PerceptionCall
from rs_hmcomm.artifacts import InMemoryArtifactStore
from rs_hmcomm.contracts import BudgetLimits, CostVector, TaskContext
from rs_hmcomm.control import (
    ContextSpec,
    ControllerObservation,
    ControllerWorkspaceSummary,
    DispatchDecision,
    EvolutionController,
    RuleRoutingPolicy,
)
from rs_hmcomm.experts import (
    EvolvingVisualExpert,
    ExpertObservation,
    ImageStore,
    VLMReasoningExpert,
    VLMVerificationExpert,
    VLMVisualExpert,
)
from rs_hmcomm.experts.mechanisms import KnowledgeExpert
from rs_hmcomm.runtime import IntegratedSingleAgentConfig, IntegratedSingleAgentRuntime
from rs_hmcomm.runtime import LangGraphMultiAgentRuntime
from rs_hmcomm.workspace import WorkspaceState, build_scoped_view


class _NoCallBackend:
    def generate_turn(self, request):  # pragma: no cover - overridden perception
        raise AssertionError("backend must not be called")


class _StubPerception(ActivePerceptionAgent):
    def __init__(self) -> None:
        super().__init__(_NoCallBackend())

    def run(self, image, question, task_type):
        boxes = (
            (0.05, 0.05, 0.15, 0.15),
            (0.30, 0.30, 0.40, 0.40),
            (0.70, 0.70, 0.80, 0.80),
        )
        return ActivePerceptionResult(
            answer="3",
            calls=(
                PerceptionCall("whole_image", "2", 10, 1, 1.0, "fake-vlm"),
                PerceptionCall(
                    "object_localization",
                    '{"objects": [{"bbox": [0.05,0.05,0.15,0.15]},'
                    '{"bbox": [0.3,0.3,0.4,0.4]},'
                    '{"bbox": [0.7,0.7,0.8,0.8]}]}',
                    10,
                    8,
                    1.0,
                    "fake-vlm",
                ),
                PerceptionCall("enhanced_global_2x", "3", 10, 1, 1.0, "fake-vlm"),
            ),
            tool_calls=1,
            decision_rule="test_multi_view_selection",
            wall_time_ms=3.0,
            candidate_boxes=boxes,
        )


def _task(task_id: str = "framework-task") -> TaskContext:
    return TaskContext(
        task_id=task_id,
        image_ref="image://framework",
        question="How many aircraft are visible?",
        task_type="object_quantity",
        answer_format="integer",
    )


def test_controller_observation_retains_only_compact_workspace_summary() -> None:
    task = _task()
    raw = WorkspaceState.initial("episode-framework", task)
    observation = ControllerObservation(task, raw, (), 0, 6)

    assert isinstance(observation.workspace, ControllerWorkspaceSummary)
    assert not isinstance(observation.workspace, WorkspaceState)
    serialized = observation.workspace.to_dict()
    assert "image_ref" not in serialized
    assert "evidence" not in serialized
    assert "artifacts" not in serialized


def test_dispatch_context_is_explicit_and_serializable() -> None:
    decision = DispatchDecision(
        "reasoning",
        "reason over selected evidence",
        ContextSpec(
            modalities=("text", "structure", "latent"),
            payload_refs=("evidence-1", "latent-1"),
            focus_node_ids=("object-1",),
            spatial_level="object",
            max_items=4,
        ),
        BudgetLimits(max_model_calls=2),
    )
    assert decision.context_spec.to_dict() == {
        "modalities": ["text", "structure", "latent"],
        "payload_refs": ["evidence-1", "latent-1"],
        "focus_node_ids": ["object-1"],
        "spatial_level": "object",
        "max_items": 4,
    }


def test_evolving_visual_expert_keeps_answer_candidates_private() -> None:
    task = _task("evolving-visual")
    image_store = ImageStore()
    image_store.put(task.image_ref, Image.new("RGB", (128, 128), "white"))
    artifact_store = InMemoryArtifactStore()
    inner_runtime = IntegratedSingleAgentRuntime(
        _StubPerception(),
        config=IntegratedSingleAgentConfig(evolution_enabled=True),
    )
    expert = EvolvingVisualExpert(
        inner_runtime,
        image_store=image_store,
        artifact_store=artifact_store,
    )
    state = WorkspaceState.initial("episode-evolving", task)
    observation = ExpertObservation(
        episode_id=state.episode_id,
        step=0,
        expert_id="visual",
        objective="inspect",
        workspace=build_scoped_view(state, ()),
    )

    result = expert.run(observation)

    assert result.workspace_delta.answer_candidates == ()
    assert result.workspace_delta.claims == ()
    assert result.workspace_delta.evidence
    assert result.workspace_delta.spatial_nodes
    assert result.workspace_delta.artifacts
    assert result.metrics["candidate_count"] >= 2
    assert result.metrics["variation_count"] >= 1
    assert result.metrics["workspace_answer_candidate_committed"] is False
    assert result.cost_delta.model_calls == 3
    assert len(artifact_store) == len(result.workspace_delta.artifacts)


class _QueueBackend:
    model_name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate_turn(self, request):
        from rs_hmcomm.backends.base import AgentOutput, BackendUsage, TokenAccounting

        return AgentOutput(
            text=self.responses.pop(0),
            usage=BackendUsage(
                input_tokens=11,
                output_tokens=3,
                token_accounting=TokenAccounting.MEASURED,
            ),
        )


def test_fake_vlm_full_collaboration_tracks_real_model_costs() -> None:
    task = _task("fake-vlm-collab")
    backend = _QueueBackend(
        [
            '{"objects": [{"label": "aircraft", "bbox": [0.1,0.1,0.2,0.2]}]}',
            "1",
            '{"verdict": "accept", "revised_answer": "1"}',
        ]
    )
    image_store = ImageStore()
    image_store.put(task.image_ref, Image.new("RGB", (64, 64), "white"))
    experts = {
        "visual": VLMVisualExpert(backend, image_store=image_store),
        "reasoning": VLMReasoningExpert(backend, image_store=image_store),
        "verification": VLMVerificationExpert(backend, image_store=image_store),
        "knowledge": KnowledgeExpert(),
    }
    runtime = LangGraphMultiAgentRuntime(
        EvolutionController(RuleRoutingPolicy()),
        experts,
        BudgetLimits(
            max_agent_calls=6,
            max_model_calls=6,
            max_model_input_tokens=1000,
            max_model_output_tokens=1000,
            max_message_text_tokens=10000,
            max_message_structure_bytes=10000,
            max_message_latent_bytes=10000,
            max_errors=1,
        ),
        max_steps=6,
    )

    result = runtime.run(task)

    assert result.answer == "1"
    assert result.agent_path == ("visual", "reasoning", "verification")
    assert result.total_cost == CostVector(
        agent_calls=3,
        model_calls=3,
        model_input_tokens=33,
        model_output_tokens=9,
        message_text_tokens=result.total_cost.message_text_tokens,
        message_structure_bytes=result.total_cost.message_structure_bytes,
    )
    assert all("context_spec" in item.decision for item in result.transitions[:3])
    result.replay()


def test_verifier_cannot_weakly_override_private_evolution_consensus() -> None:
    task = _task("conservative-verifier")
    image_store = ImageStore()
    image_store.put(task.image_ref, Image.new("RGB", (128, 128), "white"))
    artifact_store = InMemoryArtifactStore()
    visual = EvolvingVisualExpert(
        IntegratedSingleAgentRuntime(
            _StubPerception(),
            config=IntegratedSingleAgentConfig(evolution_enabled=True),
        ),
        image_store=image_store,
        artifact_store=artifact_store,
    )
    backend = _QueueBackend(
        [
            "3",
            '{"verdict": "reject", "revised_answer": "0"}',
        ]
    )
    runtime = LangGraphMultiAgentRuntime(
        EvolutionController(RuleRoutingPolicy()),
        {
            "visual": visual,
            "reasoning": VLMReasoningExpert(backend, image_store=image_store),
            "verification": VLMVerificationExpert(backend, image_store=image_store),
            "knowledge": KnowledgeExpert(),
        },
        BudgetLimits(
            max_agent_calls=6,
            max_tool_calls=6,
            max_model_calls=10,
            max_model_input_tokens=1000,
            max_model_output_tokens=1000,
            max_message_text_tokens=10000,
            max_message_structure_bytes=20000,
            max_message_latent_bytes=10000,
            max_billed_compute_ms=10000,
            max_errors=1,
        ),
        max_steps=6,
    )

    result = runtime.run(task)

    assert result.answer == "3"
    verifier = next(
        item for item in result.transitions if item.active_expert == "verification"
    )
    assert verifier.expert_metrics["verdict"] == (
        "reject_overruled_by_multiview_evidence"
    )
    assert verifier.expert_metrics["strong_multiview_support"] is True
