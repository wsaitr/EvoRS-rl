"""Offline tests for ContactSheetTool (pure geometry, no VLM)."""
from __future__ import annotations

import pytest
from PIL import Image

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.tools import ContactSheetTool, ToolContext


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
        call_id="call-cs",
        tool_name="contact_sheet",
        arguments=arguments,
    )


def _sample_detections() -> list[dict]:
    return [
        {"label": "roundabout", "bbox": [0.2, 0.2, 0.4, 0.4]},
        {"label": "roundabout", "bbox": [0.6, 0.6, 0.8, 0.8]},
        {"label": "bridge", "bbox": [0.0, 0.0, 0.3, 0.2]},
    ]


class TestContactSheetGeometry:
    def test_overview_and_panels_present(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (100, 100))
        result = tool.execute(
            _call(image_ref="input_image", detections=_sample_detections()),
            _context(image),
        )
        assert result.ok is True
        assert len(result.visuals) == 1
        sheet = result.visuals[0].image
        # Sheet is non-empty and bigger than the source (overview + 3 panels).
        assert sheet.width > 100 and sheet.height > 100

    def test_panels_metadata_numbered(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (100, 100))
        result = tool.execute(
            _call(image_ref="input_image", detections=_sample_detections()),
            _context(image),
        )
        panels = result.metadata["panels"]
        assert len(panels) == 3
        numbers = [p["number"] for p in panels]
        assert numbers == [1, 2, 3]
        labels = {p["label"] for p in panels}
        assert labels == {"roundabout", "bridge"}

    def test_empty_detections_yields_overview_only(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (100, 100))
        result = tool.execute(
            _call(image_ref="input_image", detections=[]),
            _context(image),
        )
        assert result.ok is True
        # Empty panels (frozen by normalize_json_object as a tuple).
        panels = result.metadata["panels"]
        assert len(panels) == 0
        sheet = result.visuals[0].image
        # Single overview panel only.
        assert sheet.width == 224 and sheet.height == 224

    def test_panel_count_capped_at_eight(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (100, 100))
        detections = [
            {"label": f"obj{i}", "bbox": [0.0, 0.0, 0.1, 0.1]}
            for i in range(12)
        ]
        result = tool.execute(
            _call(image_ref="input_image", detections=detections),
            _context(image),
        )
        panels = result.metadata["panels"]
        assert len(panels) == 8
        # Evidence payload records how many were dropped.
        evidence = result.evidence[0]
        assert evidence.payload["dropped_panel_count"] == 4

    def test_panel_cols_override_affects_layout(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (100, 100))
        detections = _sample_detections()
        sheet_2 = tool.execute(
            _call(image_ref="input_image", detections=detections, panel_cols=2),
            _context(image),
        ).visuals[0].image
        sheet_3 = tool.execute(
            _call(image_ref="input_image", detections=detections, panel_cols=3),
            _context(image),
        ).visuals[0].image
        # 3-column layout is wider than 2-column for the same panels.
        assert sheet_3.width > sheet_2.width

    def test_thin_boxes_produce_bounded_square_panels(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (512, 512))
        detections = [
            {"label": "windmill", "bbox": [0.10, 0.10, 0.11, 0.20]}
            for _ in range(8)
        ]
        sheet = tool.execute(
            _call(
                image_ref="input_image",
                detections=detections,
                panel_cols=2,
                panel_side=192,
            ),
            _context(image),
        ).visuals[0].image
        assert sheet.size == (384, 964)

    def test_evidence_payload_records_source_size(self):
        tool = ContactSheetTool()
        image = Image.new("RGB", (200, 150))
        result = tool.execute(
            _call(image_ref="input_image", detections=_sample_detections()),
            _context(image),
        )
        # source_size may be frozen as a tuple by normalize_json_object.
        assert tuple(result.evidence[0].payload["source_size"]) == (200, 150)


class TestContactSheetValidation:
    def test_missing_image_ref_rejected(self):
        tool = ContactSheetTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(detections=[]), _context(Image.new("RGB", (10, 10)))
            )

    def test_missing_detections_rejected(self):
        tool = ContactSheetTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="input_image"), _context(Image.new("RGB", (10, 10)))
            )

    def test_invalid_bbox_rejected(self):
        tool = ContactSheetTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(
                    image_ref="input_image",
                    detections=[{"label": "x", "bbox": [0.5, 0.5, 0.5, 0.5]}],
                ),
                _context(Image.new("RGB", (10, 10))),
            )

    def test_unknown_argument_rejected(self):
        tool = ContactSheetTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="input_image", detections=[], zoom=2),
                _context(Image.new("RGB", (10, 10))),
            )

    def test_panel_cols_out_of_range_rejected(self):
        tool = ContactSheetTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="input_image", detections=[], panel_cols=9),
                _context(Image.new("RGB", (10, 10))),
            )

    def test_unknown_image_ref_rejected(self):
        tool = ContactSheetTool()
        with pytest.raises(Exception):
            tool.execute(
                _call(image_ref="missing", detections=[]),
                _context(Image.new("RGB", (10, 10))),
            )


class TestContactSheetVisualSave:
    """Save a real contact sheet to disk for human inspection.  Not a
    functional test — only verifies the tool doesn't crash on a realistic
    detection list and produces a renderable image."""

    def test_render_realistic(self, tmp_path):
        tool = ContactSheetTool()
        # 512×512 synthetic image with a colored dot per detection.
        image = Image.new("RGB", (512, 512), (30, 30, 30))
        detections = [
            {"label": "roundabout", "bbox": [0.2, 0.6, 0.3, 0.7]},
            {"label": "roundabout", "bbox": [0.6, 0.1, 0.75, 0.2]},
            {"label": "bridge", "bbox": [0.0, 0.0, 0.4, 0.15]},
        ]
        result = tool.execute(
            _call(image_ref="input_image", detections=detections),
            _context(image),
        )
        sheet = result.visuals[0].image
        out_path = tmp_path / "contact_sheet_realistic.png"
        sheet.save(out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        # Reload to confirm a valid PNG.
        reloaded = Image.open(out_path)
        assert reloaded.size == sheet.size
