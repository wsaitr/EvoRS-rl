"""Contact-sheet tool: assemble a zoomed overview for enhanced VLM perception.

This tool is the key perceptual mechanism in EvoRS-Comm: it turns a list of
bounding boxes into a single image that a VLM can actually *see* — an
overview panel with boxes drawn on the original image, followed by zoomed
panels for each detection.  Each panel is labeled with an instance number
and semantic label, so downstream experts can refer to detections by id.

The tool is purely deterministic geometry (no VLM).  The VLM's perception
improvement comes from seeing each ROI at high resolution, not from reading
bbox coordinate text.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from PIL import Image, ImageDraw

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.contracts import Evidence
from rs_hmcomm.contracts._validation import require_id

from .base import ToolContext, ToolInputError, ToolResult
from .crop import _normalized_bbox


# Grid layout caps: avoid runaway memory on huge detection lists.
_MAX_PANELS = 8
_MIN_PANEL_SIDE = 224  # target shorter side per panel (so small objects become visible)


def _panel_cols(n: int) -> int:
    """Auto-pick a panel column count that looks reasonable."""
    if n <= 0:
        return 2
    if n <= 4:
        return 2
    if n <= 9:
        return 3
    return 4


class ContactSheetTool:
    """Build a contact sheet: overview + zoomed panels for each detection.

    Arguments:
        image_ref: id of the source visual (must be a PIL.Image.Image).
        detections: list of {label, bbox} dicts.  bbox is a normalized
            [x1, y1, x2, y2] tuple.  Up to ``_MAX_PANELS`` panels are
            rendered; extras are dropped.
        panel_cols: optional override for panel column count.

    Output: a single VisualInput containing the assembled contact sheet
    image.  The tool is deterministic — same inputs always produce the
    same output.
    """

    name = "contact_sheet"
    description = (
        "Assemble an overview + zoomed-ROI contact sheet from a list of "
        "detections.  The output image is intended to be fed to the next "
        "VLM call so it can actually see each ROI at high resolution."
    )

    def execute(self, call: CallTool, context: ToolContext) -> ToolResult:
        if call.tool_name != self.name:
            raise ToolInputError(
                f"ContactSheetTool cannot execute a '{call.tool_name}' call"
            )

        keys = set(call.arguments)
        if "image_ref" not in keys:
            raise ToolInputError("contact_sheet requires 'image_ref'")
        if "detections" not in keys:
            raise ToolInputError("contact_sheet requires 'detections'")
        unknown = keys - {"image_ref", "detections", "panel_cols", "panel_side"}
        if unknown:
            raise ToolInputError(
                f"contact_sheet unknown fields: {sorted(unknown)}"
            )

        image_ref = call.arguments["image_ref"]
        try:
            require_id(image_ref, "image_ref")
        except (TypeError, ValueError) as exc:
            raise ToolInputError(str(exc)) from exc

        detections = self._parse_detections(call.arguments["detections"])
        panel_cols = self._parse_panel_cols(call.arguments.get("panel_cols"))
        panel_side = self._parse_panel_side(call.arguments.get("panel_side"))

        try:
            source = context.visuals_by_id[image_ref]
        except KeyError as exc:
            raise ToolInputError(f"unknown image_ref: {image_ref}") from exc
        if not isinstance(source.image, Image.Image):
            raise ToolInputError("contact_sheet source must be a PIL.Image.Image")

        # Cap panels; remember which were dropped for the metadata.
        rendered = detections[:_MAX_PANELS]
        dropped = detections[_MAX_PANELS:]

        sheet = self._build_sheet(
            source.image, rendered, panel_cols=panel_cols, panel_side=panel_side
        )

        visual_id = f"visual_{call.call_id}"
        if visual_id in context.visuals_by_id:
            raise ToolInputError(
                f"contact_sheet output visual_id already exists: {visual_id}"
            )

        panels_meta = [
            {
                "number": index + 1,
                "label": label,
                "bbox": list(bbox),
            }
            for index, (label, bbox) in enumerate(rendered)
        ]
        visual = VisualInput(
            visual_id=visual_id,
            label="contact_sheet",
            image=sheet,
            source_bbox=None,
        )

        evidence = Evidence(
            evidence_id=f"evidence_{call.call_id}",
            kind="contact_sheet",
            summary=(
                f"Contact sheet with {len(rendered)} zoomed detection panel(s) "
                f"over an overview of {image_ref}."
            ),
            created_by=context.agent_id,
            created_step=context.step,
            payload={
                "tool_name": self.name,
                "call_id": call.call_id,
                "source_image_ref": image_ref,
                "source_size": list(source.image.size),
                "panels": panels_meta,
                "panel_cols": panel_cols,
                "panel_side": panel_side,
                "dropped_panel_count": len(dropped),
            },
        )

        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            ok=True,
            observation=(
                f"Contact sheet assembled; {len(rendered)} panel(s) visible "
                f"as {visual_id}."
            ),
            evidence=(evidence,),
            visuals=(visual,),
            metadata={"panels": panels_meta},
        )

    # -- internals --------------------------------------------------------

    def _build_sheet(
        self,
        source: Image.Image,
        detections: list[tuple[str, tuple[float, float, float, float]]],
        *,
        panel_cols: int,
        panel_side: int,
    ) -> Image.Image:
        source_rgb = source.convert("RGB")
        width, height = source_rgb.size

        # Overview panel: same source with boxes drawn.
        overview = source_rgb.copy()
        overview_draw = ImageDraw.Draw(overview)
        for index, (_label, bbox) in enumerate(detections):
            x1 = int(math.floor(bbox[0] * width))
            y1 = int(math.floor(bbox[1] * height))
            x2 = int(math.ceil(bbox[2] * width))
            y2 = int(math.ceil(bbox[3] * height))
            color = self._color_for(index)
            # Draw a 2-px border on the overview so small boxes remain visible.
            for offset in range(2):
                overview_draw.rectangle(
                    (x1 - offset, y1 - offset, x2 + offset, y2 + offset),
                    outline=color,
                )
            # Label each box on the overview with its panel number.
            try:
                overview_draw.text(
                    (x1 + 2, max(0, y1 - 12)),
                    f"#{index + 1}",
                    fill=color,
                )
            except Exception:
                # Font rendering may fail in some Pillow builds; boxes alone
                # are still useful.
                pass
        overview = overview.resize((panel_side, panel_side))

        # Zoomed panels: use a square context window rather than the exact
        # detection rectangle.  Remote-sensing detections are often extremely
        # thin (for example a wind-turbine mast).  Scaling the exact rectangle
        # by its shorter side creates multi-thousand-pixel strips that are both
        # expensive for the VLM and visually ambiguous.  A bounded square crop
        # preserves local context and gives every panel a predictable size.
        panels: list[Image.Image] = []
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        for index, (_label, bbox) in enumerate(detections):
            box_width = bbox[2] - bbox[0]
            box_height = bbox[3] - bbox[1]
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            side = min(1.0, max(0.10, max(box_width, box_height) * 2.5))
            left = min(max(0.0, center_x - side / 2.0), 1.0 - side)
            top = min(max(0.0, center_y - side / 2.0), 1.0 - side)
            right, bottom = left + side, top + side
            crop = source_rgb.crop(
                (
                    int(math.floor(left * width)),
                    int(math.floor(top * height)),
                    int(math.ceil(right * width)),
                    int(math.ceil(bottom * height)),
                )
            ).resize((panel_side, panel_side), resampling)

            draw = ImageDraw.Draw(crop)
            color = self._color_for(index)
            relative = (
                round((bbox[0] - left) / side * panel_side),
                round((bbox[1] - top) / side * panel_side),
                round((bbox[2] - left) / side * panel_side),
                round((bbox[3] - top) / side * panel_side),
            )
            for offset in range(3):
                draw.rectangle(
                    (
                        relative[0] - offset,
                        relative[1] - offset,
                        relative[2] + offset,
                        relative[3] + offset,
                    ),
                    outline=color,
                )
            draw.rectangle((0, 0, 30, 16), fill=(0, 0, 0))
            draw.text((3, 2), f"#{index + 1}", fill=color)
            panels.append(crop)

        # Assemble the grid.  Row 0 = overview; subsequent rows = panels.
        # Normalize each row to the same height; pad width with black.
        rows: list[list[Image.Image]] = [[overview]]
        for row_index in range(0, len(panels), panel_cols):
            rows.append(panels[row_index : row_index + panel_cols])

        row_heights = [max(cell.height for cell in row) for row in rows]
        # All rows share the same width = panel_cols * panel_side (except
        # the overview row, which uses just one cell of panel_side).
        row_widths = []
        for row_index, row in enumerate(rows):
            if row_index == 0:
                row_widths.append(panel_side)
            else:
                row_widths.append(panel_cols * panel_side)
        total_width = max(row_widths)
        total_height = sum(row_heights) + max(0, len(rows) - 1)  # 1px gap rows

        sheet = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        y_cursor = 0
        for row_index, row in enumerate(rows):
            x_cursor = 0
            for cell in row:
                # Center each cell vertically in its row band.
                y_offset = (row_heights[row_index] - cell.height) // 2
                sheet.paste(cell, (x_cursor, y_cursor + y_offset))
                x_cursor += cell.width
            y_cursor += row_heights[row_index] + 1  # +1 for gap row

        # Number labels on the overview are already drawn; labels on zoomed
        # panels are conveyed by the instance table in the metadata/prompt
        # (not by overlaying text on the pixels).
        return sheet

    @staticmethod
    def _color_for(index: int) -> tuple[int, int, int]:
        palette = [
            (255, 64, 64),
            (64, 200, 64),
            (64, 128, 255),
            (255, 200, 64),
            (200, 64, 200),
            (64, 220, 220),
            (255, 128, 0),
            (128, 255, 128),
        ]
        return palette[index % len(palette)]

    @staticmethod
    def _parse_detections(value: Any) -> list[tuple[str, tuple[float, float, float, float]]]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ToolInputError("detections must be a list or tuple")
        out: list[tuple[str, tuple[float, float, float, float]]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ToolInputError("each detection must be a mapping")
            missing = {"label", "bbox"} - set(item)
            if missing:
                raise ToolInputError(f"detection missing fields: {sorted(missing)}")
            label = str(item["label"]).strip() or "object"
            bbox = _normalized_bbox(item["bbox"])
            out.append((label, bbox))
        return out

    @staticmethod
    def _parse_panel_cols(value: Any) -> int:
        if value is None:
            return 2
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ToolInputError("panel_cols must be an integer")
        n = int(value)
        if not 1 <= n <= 4:
            raise ToolInputError("panel_cols must be within [1, 4]")
        return n

    @staticmethod
    def _parse_panel_side(value: Any) -> int:
        if value is None:
            return _MIN_PANEL_SIDE
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ToolInputError("panel_side must be an integer")
        n = int(value)
        if not 64 <= n <= 1024:
            raise ToolInputError("panel_side must be within [64, 1024]")
        return n


__all__ = ["ContactSheetTool"]
