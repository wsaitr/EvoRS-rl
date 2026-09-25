from __future__ import annotations

import json

from PIL import Image

from rs_hmcomm.agents import ActivePerceptionAgent
from rs_hmcomm.artifacts import (
    LatentFeatureConfig,
    PixelStatsLatentEncoder,
)
from rs_hmcomm.backends.base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
)
from rs_hmcomm.runtime import (
    IntegratedSingleAgentConfig,
    IntegratedSingleAgentRuntime,
)


_SIX_BOXES = [
    [0.05, 0.05, 0.10, 0.12],
    [0.20, 0.18, 0.25, 0.25],
    [0.38, 0.30, 0.43, 0.37],
    [0.55, 0.45, 0.60, 0.52],
    [0.70, 0.65, 0.75, 0.72],
    [0.85, 0.82, 0.90, 0.89],
]


class SmallObjectRecoveryBackend:
    """Direct sees two; enhanced count/localization recover six."""

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        label = request.visuals[0].label
        if "compact JSON" in request.prompt:
            boxes = _SIX_BOXES if label == "enhanced_global_2x" else _SIX_BOXES[:1]
            answer = json.dumps({"objects": [{"bbox": box} for box in boxes]})
        elif label == "enhanced_global_2x":
            answer = "7"
        else:
            answer = "2"
        return AgentOutput(
            text=answer,
            usage=BackendUsage(
                input_tokens=10,
                output_tokens=2,
                latency_ms=1.0,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="integrated-test-vlm",
            model_revision="v1",
        )


def _pixel_encoder(store):
    return PixelStatsLatentEncoder(
        store,
        LatentFeatureConfig(enabled=True),
    )


def test_integrated_episode_connects_perception_evolution_workspace_and_latent():
    runtime = IntegratedSingleAgentRuntime(
        ActivePerceptionAgent(SmallObjectRecoveryBackend()),
        config=IntegratedSingleAgentConfig(latent_enabled=True),
        latent_encoder_factory=_pixel_encoder,
    )
    result = runtime.run(
        Image.new("RGB", (128, 128), "navy"),
        "How many windmills are visible?",
        sample_id="integrated-six",
        task_type="object_quantity",
    )

    assert result.answer == "6"
    assert result.perception.answer == "6"
    assert result.perception.decision_rule == "enhanced_count_localization_consensus"
    assert result.cost.model_calls == 4
    assert result.cost.tool_calls == 1

    assert result.evolution is not None
    assert result.evolution.metrics.variations_applied >= 1
    assert result.evolution.metrics.candidates_evaluated >= 2
    assert result.evolution.selected_candidate.payload["answer"] == "6"
    private_ids = result.evolution.private_trace.candidate_ids
    assert len(private_ids) >= 2

    assert result.workspace.version == 2
    assert len(result.workspace.answer_candidates) == 1
    assert result.workspace.answer_candidates[0].answer == "6"
    assert not private_ids & result.workspace.record_ids
    levels = [node.level.value for node in result.workspace.spatial_nodes]
    assert levels.count("scene") == 1
    assert levels.count("group") == 1
    assert levels.count("object") == 6
    assert all(
        node.parent_id == "group-integrated-six"
        for node in result.workspace.spatial_nodes
        if node.level.value == "object"
    )

    assert len(result.workspace.artifacts) == 14
    assert len(result.artifact_store) == 14
    assert result.latent.enabled is True
    assert result.latent.learned_representation is False
    assert result.latent.roi_latent_count == 6
    assert result.latent.bytes_loaded == 7 * 48
    assert result.latent.value_count == 7 * 12
    assert result.latent.influenced_fitness is True
    assert result.workspace.answer_candidates[0].candidate_id in (
        result.scoped_view.visible_record_ids
    )

    payload = result.to_dict(include_private_evolution=True)
    assert payload["workspace_contains_binary_payload"] is False
    assert payload["evolution"]["private_trace"]["scope"] == (
        "private_agent_scratchpad"
    )
    workspace_json = json.dumps(payload["workspace"])
    assert "artifact_payload" not in workspace_json
    assert "pixel_values" not in workspace_json
    assert "tensor_values" not in workspace_json


def test_integrated_text_structure_path_runs_without_latent_or_evolution():
    runtime = IntegratedSingleAgentRuntime(
        ActivePerceptionAgent(SmallObjectRecoveryBackend()),
        config=IntegratedSingleAgentConfig(
            evolution_enabled=False,
            latent_enabled=False,
        ),
    )
    result = runtime.run(
        Image.new("RGB", (128, 128), "green"),
        "How many windmills are visible?",
        sample_id="structure-only",
        task_type="object_quantity",
    )

    assert result.answer == result.perception.answer == "6"
    assert result.evolution is None
    assert result.latent.enabled is False
    assert result.latent.bytes_loaded == 0
    assert len(result.workspace.artifacts) == 7
    assert all(item.kind.value != "latent" for item in result.workspace.artifacts)
    assert result.workspace.version == 2


def test_latent_flag_requires_explicit_encoder_factory():
    try:
        IntegratedSingleAgentRuntime(
            ActivePerceptionAgent(SmallObjectRecoveryBackend()),
            config=IntegratedSingleAgentConfig(latent_enabled=True),
        )
    except ValueError as exc:
        assert "latent_encoder_factory" in str(exc)
    else:  # pragma: no cover - protects the feature-flag boundary
        raise AssertionError("latent-enabled runtime accepted no encoder")


class ExistenceBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        return AgentOutput(
            text="Yes",
            usage=BackendUsage(
                input_tokens=4,
                output_tokens=1,
                latency_ms=1.0,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="existence-test-vlm",
        )


def test_existence_episode_does_not_invent_empty_hierarchy_levels():
    result = IntegratedSingleAgentRuntime(
        ActivePerceptionAgent(ExistenceBackend())
    ).run(
        Image.new("RGB", (64, 64), "gray"),
        "Is there an airport in the image?",
        sample_id="existence",
        task_type="object_existence",
    )

    assert result.answer == "Yes"
    assert result.evolution is None
    assert [node.level.value for node in result.workspace.spatial_nodes] == ["scene"]
    assert len(result.workspace.artifacts) == 1
    assert result.cost.model_calls == 1
    assert result.workspace.version == 2
