"""Grid-scan counting workflow (EXPERIMENTAL — not part of the main pipeline).

Status 2026-08-25: implemented and unit-tested at the geometry level, but
validation on real VRSBench samples showed per-tile VLM counts are too noisy
for reliable totals (see evidence/2026-08-25_zoom_counting).  The counting
weakness is a property of the quantized VLM itself, not of the tiling or
merging logic.  Kept for future revisiting with a stronger vision model;
do not wire into experiment arms.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from PIL import Image

from rs_hmcomm.backends.base import BackendRequest, ModelBackend, VisualInput


COUNT_PROMPT = (
    "This is one tile of a remote-sensing image. How many {label} "
    "(circular road junctions, parking areas, or similar) are visible?\n"
    'Respond ONLY with JSON: {{"count": <int>, "positions": [[x, y], ...]}}\n'
    "positions are each object's normalized coordinates inside this tile "
    "(origin top-left), in the same order as counted."
)

_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _parse_count_response(text: str) -> tuple[int, list[tuple[float, float]]]:
    """Extract count and positions from a VLM JSON-ish reply."""
    from .vlm_mechanisms import _extract_json_object

    parsed = _extract_json_object(text)
    if parsed is None:
        digits = re.findall(r"\d+", text)
        return (int(digits[0]) if digits else 0, [])
    try:
        count = int(parsed.get("count", 0))
    except (TypeError, ValueError):
        count = 0
    positions: list[tuple[float, float]] = []
    raw_positions = parsed.get("positions")
    if isinstance(raw_positions, list):
        for item in raw_positions:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    x = float(item[0])
                    y = float(item[1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    # Raw values kept unclamped: callers normalize pixel-canvas
                    # coordinates before any clamping makes sense.
                    positions.append((x, y))
    return (max(0, count), positions[:count])


def grid_tiles(
    rows: int,
    cols: int,
    overlap: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    """Compute overlapping tile bboxes covering [0,1]^2 (same geometry as GridCropTool)."""
    span_x = 1.0 / cols
    span_y = 1.0 / rows
    step_x = span_x * (1.0 - overlap)
    step_y = span_y * (1.0 - overlap)
    tiles: list[tuple[float, float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            x1 = min(c * step_x, 1.0 - span_x)
            y1 = min(r * step_y, 1.0 - span_y)
            if c == cols - 1:
                x1 = 1.0 - span_x
            if r == rows - 1:
                y1 = 1.0 - span_y
            tiles.append((x1, y1, x1 + span_x, y1 + span_y))
    return tiles


def exclusive_zones(
    rows: int,
    cols: int,
) -> list[tuple[float, float, float, float]]:
    """Partition [0,1]^2 into `rows*cols` disjoint cells (tile index order).

    Cell (r, c) spans x in [c/cols, (c+1)/cols), y in [r/rows, (r+1)/rows).
    """
    zones: list[tuple[float, float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            zones.append((c / cols, r / rows, (c + 1) / cols, (r + 1) / rows))
    return zones


def _zone_index(x: float, y: float, rows: int, cols: int) -> int:
    cx = min(int(x * cols), cols - 1)
    cy = min(int(y * rows), rows - 1)
    return cy * cols + cx


def merge_tile_detections(
    per_tile: list[dict[str, Any]],
    rows: int,
    cols: int,
) -> dict[str, Any]:
    """Merge per-tile counts/positions via center-ownership dedup.

    ``per_tile[i]`` = {"bbox": tile bbox, "count": n, "positions": [(x,y)…]}
    with positions in whole-image normalized coordinates.
    Returns merged unique detections with their owning tile indices.
    """
    owned: list[dict[str, Any]] = []
    for index, tile in enumerate(per_tile):
        for gx, gy in tile.get("global_positions", []):
            owner = _zone_index(gx, gy, rows, cols)
            if owner == index:
                owned.append({"x": gx, "y": gy, "tile": index})
    # Cluster near-duplicate centers (< half tile-span apart) that landed in
    # bordering zones from rounding; keep the first.
    kept: list[dict[str, Any]] = []
    min_dist = 0.5 / max(rows, cols) * 0.9
    for det in sorted(owned, key=lambda d: (d["y"], d["x"])):
        duplicate = any(
            math.hypot(det["x"] - k["x"], det["y"] - k["y"]) < min_dist
            for k in kept
        )
        if not duplicate:
            kept.append(det)
    return {
        "total": len(kept),
        "detections": kept,
        "raw_sum": sum(tile.get("count", 0) for tile in per_tile),
    }


class GridScanCounter:
    """Count question-relevant objects via tiled VLM calls + geometric merge."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        rows: int = 2,
        cols: int = 2,
        overlap: float = 0.25,
        tile_max_side: int = 448,
    ) -> None:
        self.backend = backend
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.tile_max_side = tile_max_side

    def _label_for(self, question: str) -> str:
        """Crude target extraction from a counting question."""
        match = re.search(r"how many ([a-z ]+?)\b(?:are|is|can|do|were)\b", question.lower())
        if match:
            phrase = match.group(1).strip()
            return phrase if len(phrase.split()) <= 4 else "objects of interest"
        return "objects of interest"

    def count(
        self,
        image: Image.Image,
        question: str,
    ) -> dict[str, Any]:
        label = self._label_for(question)
        width, height = image.size
        tiles = grid_tiles(self.rows, self.cols, self.overlap)

        per_tile: list[dict[str, Any]] = []
        tokens_in_total = tokens_out_total = 0
        for index, (x1, y1, x2, y2) in enumerate(tiles):
            region = image.crop((
                math.floor(x1 * width),
                math.floor(y1 * height),
                math.ceil(x2 * width),
                math.ceil(y2 * height),
            ))
            scale = self.tile_max_side / max(region.width, region.height)
            if scale < 1.0:
                region = region.resize((
                    max(28, round(region.width * scale)),
                    max(28, round(region.height * scale)),
                ))
            request = BackendRequest(
                prompt=COUNT_PROMPT.format(label=label),
                visuals=(VisualInput(
                    visual_id=f"scan-tile-{index}",
                    label=f"tile_{index}",
                    image=region,
                ),),
            )
            output = self.backend.generate_turn(request)
            usage = output.usage
            tokens_in_total += usage.input_tokens or 0
            tokens_out_total += usage.output_tokens or 0

            count, local_positions = _parse_count_response(output.text)
            # VLM reports pixel coords on the resized-tile canvas; normalize
            # by the canvas actually sent (region size after resize).
            canvas = max(region.width, region.height) or 1
            def _norm_local(lv):
                lx, ly = lv
                if max(lx, ly) > 1.0:
                    lx = min(lx / canvas, 1.0)
                    ly = min(ly / canvas, 1.0)
                return (min(max(lx, 0.0), 1.0), min(max(ly, 0.0), 1.0))
            local_positions = [_norm_local(lp) for lp in local_positions]
            global_positions = [
                (x1 + lx * (x2 - x1), y1 + ly * (y2 - y1))
                for lx, ly in local_positions
            ]
            per_tile.append({
                "tile_index": index,
                "bbox": (x1, y1, x2, y2),
                "count": count,
                "local_positions": local_positions,
                "global_positions": global_positions,
            })

        merged = merge_tile_detections(per_tile, self.rows, self.cols)
        merged["per_tile"] = [
            {
                "tile_index": t["tile_index"],
                "bbox": list(t["bbox"]),
                "count": t["count"],
                "positions": [tuple(round(v, 3) for v in p) for p in t["global_positions"]],
            }
            for t in per_tile
        ]
        merged["tokens_in"] = tokens_in_total
        merged["tokens_out"] = tokens_out_total
        merged["label"] = label
        return merged


__all__ = [
    "COUNT_PROMPT",
    "GridScanCounter",
    "_parse_count_response",
    "exclusive_zones",
    "grid_tiles",
    "merge_tile_detections",
]
