"""Safe normalized-coordinate image crop tool."""
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


NormalizedBBox = tuple[float, float, float, float]


def _normalized_bbox(value: Any) -> NormalizedBBox:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ToolInputError("bbox must be an array of four normalized numbers")
    if len(value) != 4:
        raise ToolInputError("bbox must contain exactly four values")

    numbers: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise ToolInputError("bbox values must be real numbers")
        normalized = float(coordinate)
        if not math.isfinite(normalized):
            raise ToolInputError("bbox values must be finite")
        numbers.append(normalized)

    x1, y1, x2, y2 = numbers
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ToolInputError(
            "bbox must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )
    return x1, y1, x2, y2


class CropTool:
    name = "crop"
    description = (
        "Crop a runtime image using image_ref and a normalized "
        "[x1, y1, x2, y2] bounding box."
    )

    def execute(self, call: CallTool, context: ToolContext) -> ToolResult:
        if call.tool_name != self.name:
            raise ToolInputError(
                f"CropTool cannot execute a '{call.tool_name}' call"
            )

        expected = {"image_ref", "bbox"}
        keys = set(call.arguments)
        missing = expected - keys
        unknown = keys - expected
        if missing:
            raise ToolInputError(f"crop arguments are missing: {sorted(missing)}")
        if unknown:
            raise ToolInputError(f"crop arguments contain unknown fields: {sorted(unknown)}")

        image_ref = call.arguments["image_ref"]
        try:
            require_id(image_ref, "image_ref")
        except (TypeError, ValueError) as exc:
            raise ToolInputError(str(exc)) from exc
        bbox = _normalized_bbox(call.arguments["bbox"])

        try:
            source = context.visuals_by_id[image_ref]
        except KeyError as exc:
            raise ToolInputError(f"unknown image_ref: {image_ref}") from exc
        if not isinstance(source.image, Image.Image):
            raise ToolInputError("crop source must be a PIL.Image.Image")

        width, height = source.image.size
        if width <= 0 or height <= 0:
            raise ToolInputError("crop source image must have non-zero dimensions")
        x1, y1, x2, y2 = bbox
        left = math.floor(x1 * width)
        top = math.floor(y1 * height)
        right = math.ceil(x2 * width)
        bottom = math.ceil(y2 * height)
        pixel_bbox = (left, top, right, bottom)
        if right <= left or bottom <= top:
            raise ToolInputError("bbox produces a zero-area pixel crop")

        visual_id = f"visual_{call.call_id}"
        if visual_id in context.visuals_by_id:
            raise ToolInputError(f"crop output visual_id already exists: {visual_id}")
        cropped = source.image.crop(pixel_bbox)
        visual = VisualInput(
            visual_id=visual_id,
            label=f"crop:{call.call_id}",
            image=cropped,
            source_bbox=bbox,
        )

        evidence_id = f"evidence_{call.call_id}"
        provenance = {
            "tool_name": self.name,
            "call_id": call.call_id,
            "source_image_ref": image_ref,
        }
        evidence = Evidence(
            evidence_id=evidence_id,
            kind="crop",
            summary=f"Crop extracted from {image_ref} at normalized bbox {list(bbox)}.",
            created_by=context.agent_id,
            created_step=context.step,
            artifact_refs=(visual_id,),
            payload={
                "artifact_ref": visual_id,
                "normalized_bbox": list(bbox),
                "pixel_bbox": list(pixel_bbox),
                "provenance": provenance,
            },
        )
        metadata = {
            "source_image_ref": image_ref,
            "artifact_ref": visual_id,
            "normalized_bbox": list(bbox),
            "pixel_bbox": list(pixel_bbox),
            "crop_size": [cropped.width, cropped.height],
        }
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            ok=True,
            observation=(
                f"Created {visual_id} from {image_ref}; "
                f"pixel_bbox={list(pixel_bbox)}."
            ),
            evidence=(evidence,),
            visuals=(visual,),
            metadata=metadata,
        )


__all__ = ["CropTool", "NormalizedBBox"]
