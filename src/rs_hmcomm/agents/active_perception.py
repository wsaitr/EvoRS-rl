"""Single-Agent active perception over deterministic multi-scale mosaics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
import time

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import BackendRequest, ModelBackend, VisualInput
from rs_hmcomm.tools.base import ToolContext
from rs_hmcomm.tools.contact_sheet import ContactSheetTool
from rs_hmcomm.tools.grid_crop import GridCropTool
from rs_hmcomm.tools.spatial_mosaic import SpatialMosaicTool


_COUNT_RE = re.compile(r"\b(\d{1,2})\b")


def direct_prompt(question: str, task_type: str) -> str:
    instruction = (
        "Answer with a single integer and no other text."
        if task_type == "object_quantity"
        else "Answer with exactly Yes or No and no other text."
    )
    return (
        "You are a remote-sensing image analysis assistant. "
        f"{instruction}\nQuestion: {question}\nAnswer:"
    )


def mosaic_prompt(question: str, grid_size: int) -> str:
    return (
        f"This image is one remote-sensing scene split into a {grid_size}x{grid_size} "
        "spatial mosaic. Every panel is a distinct non-overlapping region, kept "
        "in its original position and enlarged so small objects are easier to see. "
        "Inspect every panel. Count each physical object once; if an object is cut "
        "by a panel boundary, mentally join its parts rather than double-counting. "
        "Answer with a single integer and no other text.\n"
        f"Question: {question}\nAnswer:"
    )


def tile_prompt(question: str, row: int, col: int, grid_size: int) -> str:
    return (
        f"This is row {row + 1}, column {col + 1} of a {grid_size}x{grid_size} "
        "non-overlapping scan of one remote-sensing scene. The crop is enlarged. "
        "Count only the objects asked about that are visible in this crop. "
        "For an object cut by an edge, count it only when its center is inside "
        "this crop. Answer with a single integer and no other text.\n"
        f"Question: {question}\nLocal count:"
    )


def localization_prompt(question: str) -> str:
    return (
        "Inspect the remote-sensing image and locate every distinct object "
        "needed to answer the counting question. Return ONLY compact JSON in "
        'this form: {"objects":[{"bbox":[x1,y1,x2,y2]}]}. Coordinates are '
        "normalized to [0,1]. Include every visible target once and do not add "
        "uncertain background objects.\n"
        f"Question: {question}"
    )


def adjudication_prompt(
    question: str,
    *,
    whole_count: int | None,
    enhanced_count: int | None,
    localized_count: int | None,
    local_counts: list[int | None],
) -> str:
    return (
        "You are the final visual auditor for one remote-sensing scene. The "
        "image is a high-resolution enhanced view of the complete scene. Prior "
        "observations below are fallible estimates, not facts. Independently "
        "inspect the image, reject false positives, recover missed small objects, "
        "and return the exact count. Do not average estimates. Answer with a "
        "single integer and no other text.\n"
        f"Question: {question}\n"
        f"Whole-image estimate: {whole_count}\n"
        f"Enhanced-image estimate: {enhanced_count}\n"
        f"Object-localization estimate: {localized_count}\n"
        f"Non-overlapping quadrant estimates: {local_counts}\n"
        "Verified count:"
    )


def candidate_verification_prompt(question: str, candidate_index: int) -> str:
    return (
        f"This is a magnified crop around proposed object #{candidate_index}. "
        "Judge only the object nearest the crop center. Does that center object "
        "count as one of the target objects asked about? Ignore other objects and "
        "background patterns. Answer with exactly Yes or No.\n"
        f"Question: {question}\nVerdict:"
    )


def contact_sheet_audit_prompt(question: str, candidate_count: int) -> str:
    return (
        "This is a candidate contact sheet for one remote-sensing scene. The "
        "top overview contains numbered boxes; lower panels magnify those same "
        "candidate regions. Panels are not separate scenes. Some proposals may "
        "be background look-alikes or duplicate the same physical object. Inspect "
        "the overview and every zoom panel, then count only genuine distinct "
        "objects required by the question. Answer with a single integer and no "
        "other text.\n"
        f"Candidate regions: {candidate_count}\n"
        f"Question: {question}\nVerified count:"
    )


@dataclass(frozen=True, slots=True)
class PerceptionCall:
    view: str
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "answer": self.answer,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class ActivePerceptionResult:
    answer: str
    calls: tuple[PerceptionCall, ...]
    tool_calls: int
    decision_rule: str
    wall_time_ms: float
    candidate_boxes: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def model_calls(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.calls)


class ActivePerceptionAgent:
    """One VLM Agent that actively selects magnified views for counting tasks."""

    agent_id = "active-perception-generalist"

    def __init__(
        self,
        backend: ModelBackend,
        *,
        grid_sizes: tuple[int, ...] = (),
        tile_side: int = 336,
        tile_scan_size: int = 2,
    ) -> None:
        if any(size not in (2, 3, 4) for size in grid_sizes):
            raise ValueError("grid_sizes must contain values from {2, 3, 4}")
        self.backend = backend
        self.grid_sizes = tuple(grid_sizes)
        self.tile_side = tile_side
        if tile_scan_size not in (2, 3):
            raise ValueError("tile_scan_size must be 2 or 3")
        self.tile_scan_size = tile_scan_size
        self.tool = SpatialMosaicTool()
        self.grid_tool = GridCropTool()
        self.contact_sheet_tool = ContactSheetTool()

    def _ask(self, visual: VisualInput, prompt: str, view: str) -> PerceptionCall:
        output = self.backend.generate_turn(
            BackendRequest(prompt=prompt, visuals=(visual,))
        )
        usage = output.usage
        return PerceptionCall(
            view=view,
            answer=output.text.strip(),
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            latency_ms=usage.latency_ms,
            model_id=output.model_id,
        )

    @staticmethod
    def _enhance_global(image: Image.Image) -> Image.Image:
        """Preserve the whole scene while increasing effective visual resolution."""

        source = ImageOps.autocontrast(image.convert("RGB"), cutoff=0.5)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        enhanced = source.resize((source.width * 2, source.height * 2), resampling)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
        return enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

    @staticmethod
    def _enhance_tile(image: Image.Image) -> Image.Image:
        source = ImageOps.autocontrast(image.convert("RGB"), cutoff=0.5)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        enhanced = source.resize((source.width * 3, source.height * 3), resampling)
        return enhanced.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=2)
        )

    @staticmethod
    def _count(answer: str) -> int | None:
        match = _COUNT_RE.search(answer)
        return int(match.group(1)) if match else None

    @staticmethod
    def _localized_count(answer: str) -> int | None:
        boxes = ActivePerceptionAgent._localized_boxes(answer)
        return len(boxes) if boxes is not None else None

    @staticmethod
    def _strong_enhanced_consensus(
        whole_count: int | None,
        enhanced_count: int | None,
        localized_count: int | None,
    ) -> int | None:
        """Accept a recovered count only when two enhanced views support it.

        A low-resolution global view can miss tiny repeated objects.  We treat
        the enhanced free count and the enhanced instance localization as two
        independent observations.  The localization count is accepted only
        when they agree within one instance and both exceed the global count
        by a material margin.  This avoids changing an already stable global
        answer merely because localization produced one extra false proposal.
        """

        if None in (whole_count, enhanced_count, localized_count):
            return None
        assert whole_count is not None
        assert enhanced_count is not None
        assert localized_count is not None
        if (
            abs(enhanced_count - localized_count) <= 1
            and min(enhanced_count, localized_count) >= whole_count + 2
        ):
            return localized_count
        return None

    @staticmethod
    def _localized_boxes(
        answer: str,
    ) -> list[tuple[float, float, float, float]] | None:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
        raw = fenced.group(1).strip() if fenced else answer.strip()
        starts = [index for index in (raw.find("["), raw.find("{")) if index >= 0]
        if not starts:
            return None
        start = min(starts)
        closing = "]" if raw[start] == "[" else "}"
        end = raw.rfind(closing)
        if end <= start:
            return None
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        objects = payload.get("objects") if isinstance(payload, dict) else payload
        if not isinstance(objects, list):
            return None
        valid: list[tuple[float, float, float, float]] = []
        for item in objects:
            bbox = None
            if isinstance(item, dict):
                bbox = item.get("bbox", item.get("bbox_2d"))
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
            ):
                continue
            values = tuple(float(value) for value in bbox)
            scale = 1024.0 if max(values) > 1.0 else 1.0
            x1, y1, x2, y2 = (value / scale for value in values)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
            area = (x2 - x1) * (y2 - y1)
            if x1 < x2 and y1 < y2 and area <= 0.25:
                valid.append((x1, y1, x2, y2))
        return valid

    @staticmethod
    def _deduplicate_boxes(
        boxes: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        kept: list[tuple[float, float, float, float]] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            duplicate = False
            for other in kept:
                ox1, oy1, ox2, oy2 = other
                center_distance = math.hypot(
                    (x1 + x2 - ox1 - ox2) / 2.0,
                    (y1 + y2 - oy1 - oy2) / 2.0,
                )
                intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
                    0.0, min(y2, oy2) - max(y1, oy1)
                )
                other_area = (ox2 - ox1) * (oy2 - oy1)
                union = area + other_area - intersection
                if center_distance <= 0.04 or (
                    union > 0.0 and intersection / union >= 0.35
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
        return kept[:8]

    @staticmethod
    def _candidate_crop(
        image: Image.Image,
        box: tuple[float, float, float, float],
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = min(0.25, max(0.08, max(x2 - x1, y2 - y1) * 1.5))
        left = max(0.0, center_x - side / 2.0)
        top = max(0.0, center_y - side / 2.0)
        right = min(1.0, left + side)
        bottom = min(1.0, top + side)
        left, top = max(0.0, right - side), max(0.0, bottom - side)
        return image.crop(
            (
                round(left * image.width),
                round(top * image.height),
                round(right * image.width),
                round(bottom * image.height),
            )
        )

    @staticmethod
    def _map_local_box(
        box: tuple[float, float, float, float],
        tile_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = tile_bbox
        width, height = x2 - x1, y2 - y1
        return (
            x1 + box[0] * width,
            y1 + box[1] * height,
            x1 + box[2] * width,
            y1 + box[3] * height,
        )

    @staticmethod
    def _fuse_count(calls: tuple[PerceptionCall, ...]) -> tuple[str, str]:
        values = [value for item in calls if (value := ActivePerceptionAgent._count(item.answer)) is not None]
        if not values:
            return calls[0].answer, "fallback_whole_answer"
        frequencies = Counter(values)
        value, frequency = max(
            frequencies.items(),
            key=lambda item: (item[1], -abs(item[0] - values[0]), -item[0]),
        )
        if frequency >= 2:
            return str(value), "multi_view_majority"
        ordered = sorted(values)
        return str(ordered[len(ordered) // 2]), "multi_view_median"

    def run(
        self,
        image: Image.Image,
        question: str,
        task_type: str,
    ) -> ActivePerceptionResult:
        if task_type not in {"object_quantity", "object_existence"}:
            raise ValueError(f"unsupported task_type: {task_type}")
        started = time.perf_counter()
        whole = VisualInput("input_image", "whole_image", image)
        calls: list[PerceptionCall] = [
            self._ask(whole, direct_prompt(question, task_type), "whole_image")
        ]
        if task_type == "object_existence":
            return ActivePerceptionResult(
                answer=calls[0].answer,
                calls=tuple(calls),
                tool_calls=0,
                decision_rule="whole_image_sufficient",
                wall_time_ms=(time.perf_counter() - started) * 1000.0,
            )

        localization = self._ask(
            whole,
            localization_prompt(question),
            "object_localization",
        )
        calls.append(localization)
        enhanced_image = self._enhance_global(image)
        enhanced_visual = VisualInput(
            "enhanced_global",
            "enhanced_global_2x",
            enhanced_image,
        )
        enhanced_call = self._ask(
            enhanced_visual,
            direct_prompt(question, task_type),
            "enhanced_global_2x",
        )
        calls.append(enhanced_call)
        whole_count = self._count(calls[0].answer)
        enhanced_count = self._count(enhanced_call.answer)
        if whole_count is not None and whole_count == enhanced_count:
            return ActivePerceptionResult(
                answer=str(whole_count),
                calls=tuple(calls),
                tool_calls=1,
                decision_rule="global_scale_consensus",
                wall_time_ms=(time.perf_counter() - started) * 1000.0,
                candidate_boxes=tuple(self._localized_boxes(localization.answer) or ()),
            )
        enhanced_localization = self._ask(
            enhanced_visual,
            localization_prompt(question),
            "enhanced_object_localization",
        )
        calls.append(enhanced_localization)

        original_boxes = self._localized_boxes(localization.answer) or []
        enhanced_boxes = self._localized_boxes(enhanced_localization.answer) or []
        recovered_count = self._strong_enhanced_consensus(
            whole_count,
            enhanced_count,
            len(enhanced_boxes),
        )
        if recovered_count is not None:
            return ActivePerceptionResult(
                answer=str(recovered_count),
                calls=tuple(calls),
                tool_calls=1,
                decision_rule="enhanced_count_localization_consensus",
                wall_time_ms=(time.perf_counter() - started) * 1000.0,
                candidate_boxes=tuple(enhanced_boxes),
            )
        localization_grid = self.grid_tool.execute(
            CallTool(
                call_id="localization-grid-2x2",
                tool_name=self.grid_tool.name,
                arguments={
                    "image_ref": whole.visual_id,
                    "rows": 2,
                    "cols": 2,
                    "overlap": 0.0,
                },
            ),
            ToolContext(
                agent_id=self.agent_id,
                step=3,
                visuals_by_id={whole.visual_id: whole},
            ),
        )
        tiled_boxes: list[tuple[float, float, float, float]] = []
        for tile in localization_grid.visuals:
            tile_localization = self._ask(
                VisualInput(
                    f"enhanced_{tile.visual_id}",
                    f"enhanced_{tile.label}_3x",
                    self._enhance_tile(tile.image),
                    source_bbox=tile.source_bbox,
                ),
                localization_prompt(question),
                f"tile_localization_{tile.label}",
            )
            calls.append(tile_localization)
            if tile.source_bbox is not None:
                tiled_boxes.extend(
                    self._map_local_box(box, tile.source_bbox)
                    for box in (self._localized_boxes(tile_localization.answer) or [])
                )
        candidate_boxes = self._deduplicate_boxes(
            original_boxes + enhanced_boxes + tiled_boxes
        )
        if candidate_boxes:
            contact_call = CallTool(
                call_id="candidate-contact-sheet",
                tool_name=self.contact_sheet_tool.name,
                arguments={
                    "image_ref": whole.visual_id,
                    "detections": [
                        {"label": "candidate", "bbox": list(box)}
                        for box in candidate_boxes
                    ],
                    "panel_cols": 2,
                    "panel_side": 192,
                },
            )
            contact_result = self.contact_sheet_tool.execute(
                contact_call,
                ToolContext(
                    agent_id=self.agent_id,
                    step=3,
                    visuals_by_id={whole.visual_id: whole},
                ),
            )
            contact_audit = self._ask(
                contact_result.visuals[0],
                contact_sheet_audit_prompt(question, len(candidate_boxes)),
                "candidate_contact_sheet",
            )
            calls.append(contact_audit)
            audited_count = self._count(contact_audit.answer)
            if audited_count is not None:
                return ActivePerceptionResult(
                    answer=str(audited_count),
                    calls=tuple(calls),
                    tool_calls=2 + len(localization_grid.visuals),
                    decision_rule="candidate_contact_sheet_audit",
                    wall_time_ms=(time.perf_counter() - started) * 1000.0,
                    candidate_boxes=tuple(candidate_boxes),
                )

        for index, grid_size in enumerate(self.grid_sizes, start=1):
            call = CallTool(
                call_id=f"mosaic-{grid_size}x{grid_size}-{index}",
                tool_name=self.tool.name,
                arguments={
                    "image_ref": whole.visual_id,
                    "grid_size": grid_size,
                    "tile_side": self.tile_side,
                    "gap": 2,
                },
            )
            tool_result = self.tool.execute(
                call,
                ToolContext(
                    agent_id=self.agent_id,
                    step=index,
                    visuals_by_id={whole.visual_id: whole},
                ),
            )
            calls.append(
                self._ask(
                    tool_result.visuals[0],
                    mosaic_prompt(question, grid_size),
                    f"mosaic_{grid_size}x{grid_size}",
                )
            )

        grid_call = CallTool(
            call_id=f"tile-scan-{self.tile_scan_size}x{self.tile_scan_size}",
            tool_name=self.grid_tool.name,
            arguments={
                "image_ref": whole.visual_id,
                "rows": self.tile_scan_size,
                "cols": self.tile_scan_size,
                "overlap": 0.0,
            },
        )
        grid_result = self.grid_tool.execute(
            grid_call,
            ToolContext(
                agent_id=self.agent_id,
                step=len(self.grid_sizes) + 1,
                visuals_by_id={whole.visual_id: whole},
            ),
        )
        tile_calls: list[PerceptionCall] = []
        for visual in grid_result.visuals:
            row_col = re.search(r"tile_r(\d+)c(\d+)", visual.label)
            if row_col is None:
                raise RuntimeError("grid tool returned an invalid tile label")
            row, col = (int(row_col.group(1)), int(row_col.group(2)))
            tile_calls.append(
                self._ask(
                    VisualInput(
                        f"enhanced_{visual.visual_id}",
                        f"enhanced_{visual.label}_3x",
                        self._enhance_tile(visual.image),
                        source_bbox=visual.source_bbox,
                    ),
                    tile_prompt(question, row, col, self.tile_scan_size),
                    visual.label,
                )
            )
        calls.extend(tile_calls)
        local_counts = [self._count(item.answer) for item in tile_calls]
        if all(value is not None for value in local_counts):
            tile_sum = sum(value for value in local_counts if value is not None)
            whole_count = self._count(calls[0].answer)
            enhanced_count = self._count(enhanced_call.answer)
            localized_count = self._localized_count(localization.answer)
            count_candidates = [
                value
                for value in (
                    whole_count,
                    enhanced_count,
                    tile_sum,
                    localized_count,
                )
                if value is not None
            ]
            frequencies = Counter(count_candidates)
            selected, frequency = max(
                frequencies.items(), key=lambda item: (item[1], -item[0])
            )
            if frequency >= 2:
                answer = str(selected)
                rule = "multi_view_perception_majority"
            elif enhanced_count is not None:
                answer = str(enhanced_count)
                rule = "enhanced_global_tiebreak"
            else:
                answer = str(tile_sum)
                rule = f"tile_sum_{self.tile_scan_size}x{self.tile_scan_size}"
        else:
            answer, rule = self._fuse_count(tuple(calls[: 1 + len(self.grid_sizes)]))

        adjudication = self._ask(
            VisualInput(
                "enhanced_global_audit",
                "enhanced_global_2x",
                self._enhance_global(image),
            ),
            adjudication_prompt(
                question,
                whole_count=self._count(calls[0].answer),
                enhanced_count=self._count(enhanced_call.answer),
                localized_count=self._localized_count(localization.answer),
                local_counts=local_counts,
            ),
            "visual_adjudication",
        )
        calls.append(adjudication)
        audited_count = self._count(adjudication.answer)
        if audited_count is not None:
            answer = str(audited_count)
            rule = "visual_adjudication"
        return ActivePerceptionResult(
            answer=answer,
            calls=tuple(calls),
            tool_calls=len(self.grid_sizes) + 2,
            decision_rule=rule,
            wall_time_ms=(time.perf_counter() - started) * 1000.0,
            candidate_boxes=tuple(candidate_boxes),
        )


__all__ = [
    "ActivePerceptionAgent",
    "ActivePerceptionResult",
    "PerceptionCall",
    "direct_prompt",
    "mosaic_prompt",
    "localization_prompt",
    "adjudication_prompt",
    "candidate_verification_prompt",
    "contact_sheet_audit_prompt",
    "tile_prompt",
]
