"""Deterministic grid-crop tool: slice an image into overlapping tiles.

Unlike the model-driven detection path, this tool is pure geometry: given a
normalized region and a grid size it always returns the same tiles.  It gives
the visual expert a reliable zoom-in primitive — each tile can then be
re-examined by the VLM at higher effective resolution.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

from PIL import Image

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.contracts import Evidence
from rs_hmcomm.contracts._validation import require_id

from .base import ToolContext, ToolInputError, ToolResult
from .crop import _normalized_bbox


class GridCropTool:
    """Slice a normalized region into an N×M grid of overlapping tiles.

    Arguments:
        image_ref: id of the source visual (usually "input_image").
        bbox: optional normalized [x1, y1, x2, y2] region to slice
              (defaults to the whole image).
        rows / cols: grid dimensions (default 2×2).  Values 1-4 are allowed;
              larger grids produce tiles too small for the vision tower.
        overlap: fractional overlap between adjacent tiles (default 0.15).

    Outputs one VisualInput per tile (visual_id ``grid_<call>_<r><c>``) plus
    a JSON observation listing each tile's normalized bbox so the agent can
    reference tiles by position later.
    """

    name = "grid_crop"
    description = (
        "Slice an image (or a bbox region) into an overlapping rows×cols "
        "grid of zoomed tiles for closer inspection."
    )

    def execute(self, call: CallTool, context: ToolContext) -> ToolResult:
        if call.tool_name != self.name:
            raise ToolInputError(f"GridCropTool cannot execute a '{call.tool_name}' call")

        expected = {"image_ref", "bbox", "rows", "cols", "overlap"}
        keys = set(call.arguments)
        missing = keys & {"image_ref"} ^ {"image_ref"}
        if missing:
            raise ToolInputError(f"grid_crop arguments are missing: {sorted(missing)}")
        unknown = keys - expected
        if unknown:
            raise ToolInputError(
                f"grid_crop arguments contain unknown fields: {sorted(unknown)}"
            )

        image_ref = call.arguments["image_ref"]
        try:
            require_id(image_ref, "image_ref")
        except (TypeError, ValueError) as exc:
            raise ToolInputError(str(exc)) from exc

        bbox_value = call.arguments.get("bbox")
        bbox = _normalized_bbox(bbox_value) if bbox_value is not None else (0.0, 0.0, 1.0, 1.0)

        rows = self._positive_small_int(call.arguments.get("rows", 2), "rows")
        cols = self._positive_small_int(call.arguments.get("cols", 2), "cols")
        overlap = self._fraction(call.arguments.get("overlap", 0.15), "overlap")

        try:
            source = context.visuals_by_id[image_ref]
        except KeyError as exc:
            raise ToolInputError(f"unknown image_ref: {image_ref}") from exc
        if not isinstance(source.image, Image.Image):
            raise ToolInputError("grid_crop source must be a PIL.Image.Image")

        width, height = source.image.size
        x1, y1, x2, y2 = bbox
        region_px = (
            math.floor(x1 * width),
            math.floor(y1 * height),
            math.ceil(x2 * width),
            math.ceil(y2 * height),
        )
        region = source.image.crop(region_px)
        region_w = region_px[2] - region_px[0]
        region_h = region_px[3] - region_px[1]
        if region_w <= 0 or region_h <= 0:
            raise ToolInputError("bbox produces a zero-area pixel region")

        # Tile geometry in region-local normalized coordinates.
        span_x = 1.0 / cols
        span_y = 1.0 / rows
        step_x = span_x * (1.0 - overlap)
        step_y = span_y * (1.0 - overlap)

        visuals: list[VisualInput] = []
        tile_records: list[dict[str, Any]] = []
        for r in range(rows):
            for c in range(cols):
                local_x1 = min(c * step_x, 1.0 - span_x)
                local_y1 = min(r * step_y, 1.0 - span_y)
                # The last row/column aligns flush with the region edge so
                # the tiles collectively cover the full region.
                if c == cols - 1:
                    local_x1 = 1.0 - span_x
                if r == rows - 1:
                    local_y1 = 1.0 - span_y
                local_x2 = local_x1 + span_x
                local_y2 = local_y1 + span_y
                tile = region.crop((
                    round(local_x1 * region_w),
                    round(local_y1 * region_h),
                    round(local_x2 * region_w),
                    round(local_y2 * region_h),
                ))
                # Map tile back to whole-image normalized coordinates.
                global_x1 = x1 + local_x1 * (x2 - x1)
                global_x2 = x1 + local_x2 * (x2 - x1)
                global_y1 = y1 + local_y1 * (y2 - y1)
                global_y2 = y1 + local_y2 * (y2 - y1)

                visual_id = f"grid_{call.call_id}_{r}{c}"
                if visual_id in context.visuals_by_id:
                    raise ToolInputError(
                        f"grid_crop output visual_id already exists: {visual_id}"
                    )
                visuals.append(
                    VisualInput(
                        visual_id=visual_id,
                        label=f"tile_r{r}c{c}",
                        image=tile,
                        source_bbox=(global_x1, global_y1, global_x2, global_y2),
                    )
                )
                tile_records.append(
                    {
                        "visual_id": visual_id,
                        "row": r,
                        "col": c,
                        "global_bbox": [
                            round(global_x1, 4),
                            round(global_y1, 4),
                            round(global_x2, 4),
                            round(global_y2, 4),
                        ],
                    }
                )

        evidence = Evidence(
            evidence_id=f"evidence_{call.call_id}",
            kind="grid_crop",
            summary=(
                f"Grid crop of {image_ref} into {rows}x{cols} overlapping tiles "
                f"(overlap={overlap})."
            ),
            created_by=context.agent_id,
            created_step=context.step,
            payload={
                "tool_name": self.name,
                "call_id": call.call_id,
                "source_image_ref": image_ref,
                "region_bbox": list(bbox),
                "tiles": tile_records,
            },
        )

        observation = {
            "grid": f"{rows}x{cols}",
            "overlap": overlap,
            "region_bbox": list(bbox),
            "tiles": tile_records,
        }
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            ok=True,
            observation=(
                f"Sliced into {rows}x{cols} tiles; inspect each tile visual "
                f"by its visual_id."
            ),
            evidence=(evidence,),
            visuals=tuple(visuals),
            metadata={"observation": observation},
        )

    @staticmethod
    def _positive_small_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ToolInputError(f"{name} must be an integer")
        number = int(value)
        if not 1 <= number <= 4:
            raise ToolInputError(f"{name} must be within [1, 4]")
        return number

    @staticmethod
    def _fraction(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ToolInputError(f"{name} must be a number")
        number = float(value)
        if not 0.0 <= number < 0.5:
            raise ToolInputError(f"{name} must be within [0, 0.5)")
        return number


__all__ = ["GridCropTool"]
