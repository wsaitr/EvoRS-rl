from PIL import Image
import pytest

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.tools import SpatialMosaicTool, ToolContext, ToolInputError


def _quadrants() -> Image.Image:
    image = Image.new("RGB", (8, 8))
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for row in range(2):
        for col in range(2):
            panel = Image.new("RGB", (4, 4), colors[row * 2 + col])
            image.paste(panel, (col * 4, row * 4))
    return image


def test_spatial_mosaic_preserves_non_overlapping_layout():
    tool = SpatialMosaicTool()
    result = tool.execute(
        CallTool(
            call_id="mosaic-test",
            tool_name="spatial_mosaic",
            arguments={
                "image_ref": "image",
                "grid_size": 2,
                "tile_side": 112,
                "gap": 2,
            },
        ),
        ToolContext(
            agent_id="generalist",
            step=1,
            visuals_by_id={"image": VisualInput("image", "scene", _quadrants())},
        ),
    )
    mosaic = result.visuals[0].image
    assert mosaic.size == (226, 226)
    assert mosaic.getpixel((56, 56))[0] > 240
    assert mosaic.getpixel((170, 56))[1] > 240
    assert mosaic.getpixel((56, 170))[2] > 240
    assert mosaic.getpixel((170, 170))[:2] == (255, 255)
    assert result.metadata["content_overlap"] is False
    assert result.evidence[0].payload["source_size"] == (8, 8)


def test_spatial_mosaic_rejects_invalid_grid():
    tool = SpatialMosaicTool()
    with pytest.raises(ToolInputError, match="grid_size"):
        tool.execute(
            CallTool(
                call_id="bad-grid",
                tool_name="spatial_mosaic",
                arguments={"image_ref": "image", "grid_size": 1},
            ),
            ToolContext(
                agent_id="generalist",
                step=0,
                visuals_by_id={"image": VisualInput("image", "scene", _quadrants())},
            ),
        )
