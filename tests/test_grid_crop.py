"""Offline tests for GridCropTool (pure geometry, no model calls)."""
from __future__ import annotations

import pytest
from PIL import Image

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.tools import GridCropTool, ToolContext


def _context(image: Image.Image, agent_id="agent-x", step=0) -> ToolContext:
    return ToolContext(
        agent_id=agent_id,
        step=step,
        visuals_by_id={
            "input_image": VisualInput(
                visual_id="input_image", label="img", image=image
            )
        },
    )


def _call(**arguments) -> CallTool:
    return CallTool(
        call_id="call-1",
        tool_name="grid_crop",
        arguments=arguments,
    )


class TestGridCropGeometry:
    def test_default_2x2_covers_whole_image(self):
        tool = GridCropTool()
        image = Image.new("RGB", (100, 100))
        result = tool.execute(_call(image_ref="input_image"), _context(image))
        assert result.ok is True
        tiles = result.metadata["observation"]["tiles"]
        assert len(tiles) == 4
        # Union of tile bboxes covers the full image.
        xs1 = min(t["global_bbox"][0] for t in tiles)
        ys1 = min(t["global_bbox"][1] for t in tiles)
        xs2 = max(t["global_bbox"][2] for t in tiles)
        ys2 = max(t["global_bbox"][3] for t in tiles)
        assert (xs1, ys1, xs2, ys2) == (0.0, 0.0, 1.0, 1.0)

    def test_tile_pixel_sizes_match_region(self):
        tool = GridCropTool()
        image = Image.new("RGB", (200, 100))
        result = tool.execute(
            _call(image_ref="input_image", rows=2, cols=1), _context(image)
        )
        tiles = [v for v in result.visuals]
        assert len(tiles) == 2
        # Each tile is half the height of the 200x100 source (with overlap).
        heights = sorted(v.image.size[1] for v in tiles)
        assert all(40 <= h <= 60 for h in heights)
        assert all(v.image.size[0] == 200 for v in tiles)

    def test_region_bbox_limits_tiles(self):
        tool = GridCropTool()
        image = Image.new("RGB", (100, 100))
        result = tool.execute(
            _call(image_ref="input_image", bbox=[0.25, 0.25, 0.75, 0.75], rows=2, cols=2),
            _context(image),
        )
        tiles = result.metadata["observation"]["tiles"]
        for tile in tiles:
            x1, y1, x2, y2 = tile["global_bbox"]
            assert 0.25 <= x1 < x2 <= 0.75
            assert 0.25 <= y1 < y2 <= 0.75

    def test_overlap_reduces_tile_step(self):
        tool = GridCropTool()
        image = Image.new("RGB", (100, 100))
        no_overlap = tool.execute(
            _call(image_ref="input_image", rows=1, cols=3, overlap=0.0), _context(image)
        )
        with_overlap = tool.execute(
            _call(image_ref="input_image", rows=1, cols=3, overlap=0.3), _context(image)
        )
        step_no = (
            no_overlap.metadata["observation"]["tiles"][1]["global_bbox"][0]
            - no_overlap.metadata["observation"]["tiles"][0]["global_bbox"][0]
        )
        step_with = (
            with_overlap.metadata["observation"]["tiles"][1]["global_bbox"][0]
            - with_overlap.metadata["observation"]["tiles"][0]["global_bbox"][0]
        )
        # Overlap shrinks the distance between adjacent tile origins...
        assert step_with < step_no
        # ...while each tile keeps the same span, so coverage overlaps.
        span_no = (
            no_overlap.metadata["observation"]["tiles"][0]["global_bbox"][2]
            - no_overlap.metadata["observation"]["tiles"][0]["global_bbox"][0]
        )
        span_with = (
            with_overlap.metadata["observation"]["tiles"][0]["global_bbox"][2]
            - with_overlap.metadata["observation"]["tiles"][0]["global_bbox"][0]
        )
        assert abs(span_no - span_with) < 1e-9

    def test_source_bbox_recorded_on_visuals(self):
        tool = GridCropTool()
        image = Image.new("RGB", (100, 100))
        result = tool.execute(_call(image_ref="input_image"), _context(image))
        for visual in result.visuals:
            assert visual.source_bbox is not None
            x1, y1, x2, y2 = visual.source_bbox
            assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


class TestGridCropValidation:
    def test_rows_out_of_range_rejected(self):
        tool = GridCropTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="input_image", rows=5), _context(Image.new("RGB", (10, 10)))
            )

    def test_bad_overlap_rejected(self):
        tool = GridCropTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="input_image", overlap=0.9),
                _context(Image.new("RGB", (10, 10))),
            )

    def test_unknown_image_ref_rejected(self):
        tool = GridCropTool()
        with pytest.raises(Exception):
            tool.execute(_call(image_ref="missing"), _context(Image.new("RGB", (10, 10))))

    def test_unknown_argument_rejected(self):
        tool = GridCropTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="input_image", zoom=8), _context(Image.new("RGB", (10, 10)))
            )
