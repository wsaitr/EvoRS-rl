from __future__ import annotations

import json
import math
from types import SimpleNamespace

from PIL import Image
import pytest

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.tools import (
    CropTool,
    DuplicateToolError,
    ToolContext,
    ToolInputError,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)


def _context(image: object | None = None) -> ToolContext:
    source = VisualInput(
        visual_id="input_image",
        label="original",
        image=image if image is not None else Image.new("RGB", (100, 80), "white"),
    )
    return ToolContext(
        agent_id="generalist",
        step=1,
        visuals_by_id={source.visual_id: source},
    )


def _call(
    *,
    image_ref: str = "input_image",
    bbox: object = (0.25, 0.25, 0.75, 0.75),
) -> CallTool:
    return CallTool(
        call_id="crop_1",
        tool_name="crop",
        arguments={"image_ref": image_ref, "bbox": bbox},
    )


def test_crop_tool_returns_pixel_bbox_visual_and_grounded_evidence() -> None:
    source = Image.new("RGB", (100, 80), "white")
    result = CropTool().execute(_call(), _context(source))

    assert result.ok is True
    assert tuple(result.metadata["pixel_bbox"]) == (25, 20, 75, 60)
    assert tuple(result.metadata["crop_size"]) == (50, 40)
    assert len(result.visuals) == 1
    assert result.visuals[0].visual_id == "visual_crop_1"
    assert result.visuals[0].source_bbox == (0.25, 0.25, 0.75, 0.75)
    assert isinstance(result.visuals[0].image, Image.Image)
    assert result.visuals[0].image.size == (50, 40)

    evidence = result.evidence[0]
    assert evidence.created_by == "generalist"
    assert evidence.created_step == 1
    assert evidence.artifact_refs == ("visual_crop_1",)
    assert evidence.payload["artifact_ref"] == "visual_crop_1"
    assert tuple(evidence.payload["pixel_bbox"]) == (25, 20, 75, 60)
    assert evidence.payload["provenance"] == {
        "tool_name": "crop",
        "call_id": "crop_1",
        "source_image_ref": "input_image",
    }
    assert source.size == (100, 80)


def test_tool_result_log_view_is_json_safe_and_excludes_image_objects() -> None:
    result = CropTool().execute(_call(), _context())

    encoded = json.dumps(result.to_log_dict(), allow_nan=False)

    assert "visual_crop_1" in encoded
    assert "PIL" not in encoded
    assert '"image"' not in encoded


def test_tool_result_thaws_nested_metadata_for_json_logging() -> None:
    result = ToolResult(
        call_id="call_1",
        tool_name="test_tool",
        ok=True,
        observation="done",
        metadata={"nested": {"coords": [1, 2, 3]}},
    )

    assert json.dumps(result.to_log_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "bbox",
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [-0.1, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.1, 1.0],
        [0.8, 0.0, 0.2, 1.0],
        [0.0, 0.8, 1.0, 0.2],
        [0.0, 0.0, 1.0],
        "0,0,1,1",
        [False, 0.0, 1.0, 1.0],
    ],
)
def test_crop_tool_rejects_invalid_bbox(bbox: object) -> None:
    with pytest.raises(ToolInputError):
        CropTool().execute(_call(bbox=bbox), _context())


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_crop_tool_itself_rejects_non_finite_bbox(coordinate: float) -> None:
    # Strict CallTool construction rejects these values even earlier.  This
    # intentionally bypasses that boundary to verify CropTool is safe when
    # called by another runtime implementation.
    unsafe_call = SimpleNamespace(
        call_id="crop_unsafe",
        tool_name="crop",
        arguments={
            "image_ref": "input_image",
            "bbox": [0.0, 0.0, coordinate, 1.0],
        },
    )
    with pytest.raises(ToolInputError, match="finite"):
        CropTool().execute(unsafe_call, _context())  # type: ignore[arg-type]


def test_crop_tool_rejects_non_pil_source() -> None:
    with pytest.raises(ToolInputError, match="PIL"):
        CropTool().execute(_call(), _context({"pixels": []}))


def test_crop_tool_rejects_unknown_image_ref() -> None:
    with pytest.raises(ToolInputError, match="unknown image_ref"):
        CropTool().execute(_call(image_ref="missing_image"), _context())


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"image_ref": "input_image"}, "missing"),
        (
            {
                "image_ref": "input_image",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "path": "/tmp/untrusted",
            },
            "unknown",
        ),
    ],
)
def test_crop_tool_rejects_missing_or_unknown_arguments(
    arguments: dict[str, object],
    message: str,
) -> None:
    call = CallTool("crop_1", "crop", arguments)
    with pytest.raises(ToolInputError, match=message):
        CropTool().execute(call, _context())


def test_crop_tool_rejects_output_visual_id_collision() -> None:
    source = VisualInput("input_image", "original", Image.new("RGB", (10, 10)))
    collision = VisualInput("visual_crop_1", "existing", Image.new("RGB", (2, 2)))
    context = ToolContext(
        agent_id="generalist",
        step=0,
        visuals_by_id={
            source.visual_id: source,
            collision.visual_id: collision,
        },
    )
    with pytest.raises(ToolInputError, match="already exists"):
        CropTool().execute(_call(), context)


def test_tool_registry_is_an_explicit_allow_list() -> None:
    tool = CropTool()
    registry = ToolRegistry([tool])

    assert registry.names == ("crop",)
    assert registry.get("crop") is tool
    assert registry.execute(_call(), _context()).ok
    assert json.dumps(registry.descriptions(), allow_nan=False)

    with pytest.raises(DuplicateToolError):
        registry.register(CropTool())
    with pytest.raises(UnknownToolError):
        registry.get("not_registered")


def test_tool_context_takes_a_read_only_mapping_snapshot() -> None:
    source = VisualInput("input_image", "original", Image.new("RGB", (10, 10)))
    original = {source.visual_id: source}
    context = ToolContext("generalist", 0, original)
    original.clear()

    assert context.visuals_by_id["input_image"] is source
    with pytest.raises(TypeError):
        context.visuals_by_id["another"] = source  # type: ignore[index]
