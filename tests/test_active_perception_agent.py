from PIL import Image

from rs_hmcomm.agents.active_perception import ActivePerceptionAgent
from rs_hmcomm.backends.base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
)


class ViewAwareBackend:
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        label = request.visuals[0].label
        answer = "3" if label.startswith("spatial_mosaic") else "2"
        if "compact JSON" in request.prompt:
            answer = '{"objects":[{"bbox":[0.1,0.1,0.2,0.2]},{"bbox":[0.6,0.6,0.7,0.7]}]}'
        if label == "enhanced_global_2x":
            answer = "4"
        if label == "contact_sheet":
            answer = "2"
        if label.startswith("enhanced_tile_"):
            answer = "1"
        if "Yes or No" in request.prompt:
            answer = "Yes"
        return AgentOutput(
            text=answer,
            usage=BackendUsage(
                input_tokens=10,
                output_tokens=1,
                latency_ms=2.0,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id="test-vlm",
        )


def test_counting_uses_real_mosaics_and_majority():
    agent = ActivePerceptionAgent(ViewAwareBackend(), grid_sizes=(2, 3), tile_side=112)
    result = agent.run(
        Image.new("RGB", (32, 32), "white"),
        "How many aircraft are visible?",
        "object_quantity",
    )
    assert result.answer == "2"
    assert result.decision_rule == "candidate_contact_sheet_audit"
    assert result.model_calls == 9
    assert result.tool_calls == 6
    assert [item.view for item in result.calls] == [
        "whole_image",
        "object_localization",
        "enhanced_global_2x",
        "enhanced_object_localization",
        "tile_localization_tile_r0c0",
        "tile_localization_tile_r0c1",
        "tile_localization_tile_r1c0",
        "tile_localization_tile_r1c1",
        "candidate_contact_sheet",
    ]
    assert result.input_tokens == 90


def test_existence_keeps_global_scene_without_unneeded_tools():
    agent = ActivePerceptionAgent(ViewAwareBackend())
    result = agent.run(
        Image.new("RGB", (32, 32), "white"),
        "Is there a road?",
        "object_existence",
    )
    assert result.answer == "Yes"
    assert result.model_calls == 1
    assert result.tool_calls == 0
    assert result.decision_rule == "whole_image_sufficient"


def test_three_way_disagreement_uses_median_count():
    calls = tuple(
        type("Call", (), {"answer": value})()
        for value in ("1", "5", "3")
    )
    answer, rule = ActivePerceptionAgent._fuse_count(calls)  # type: ignore[arg-type]
    assert answer == "3"
    assert rule == "multi_view_median"


def test_strong_enhanced_consensus_recovers_small_objects():
    assert ActivePerceptionAgent._strong_enhanced_consensus(2, 7, 6) == 6
    assert ActivePerceptionAgent._strong_enhanced_consensus(2, 2, 3) is None
    assert ActivePerceptionAgent._strong_enhanced_consensus(2, 4, 2) is None


class ScaleStableBackend(ViewAwareBackend):
    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        output = super().generate_turn(request)
        if request.visuals[0].label == "enhanced_global_2x":
            return AgentOutput(
                text="2",
                usage=output.usage,
                model_id=output.model_id,
            )
        return output


def test_matching_global_scales_stop_before_expensive_local_scan():
    result = ActivePerceptionAgent(ScaleStableBackend()).run(
        Image.new("RGB", (32, 32), "white"),
        "How many aircraft are visible?",
        "object_quantity",
    )
    assert result.answer == "2"
    assert result.decision_rule == "global_scale_consensus"
    assert result.model_calls == 3
    assert result.tool_calls == 1
