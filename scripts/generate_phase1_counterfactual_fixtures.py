"""Generate deterministic image-sensitive fixtures for the Phase 1 live gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


OUTPUT_ROOT = Path("tests/fixtures/phase1")
QUESTION = "How many white aircraft silhouettes are visible on the apron?"


def _draw_aircraft(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    # One connected, high-contrast top-down aircraft silhouette.  No textual
    # digit is embedded in the image, so the answer must come from vision.
    points = (
        (x, y - 25),
        (x + 5, y - 10),
        (x + 22, y - 3),
        (x + 22, y + 3),
        (x + 5, y),
        (x + 5, y + 15),
        (x + 12, y + 22),
        (x + 12, y + 25),
        (x, y + 21),
        (x - 12, y + 25),
        (x - 12, y + 22),
        (x - 5, y + 15),
        (x - 5, y),
        (x - 22, y + 3),
        (x - 22, y - 3),
        (x - 5, y - 10),
    )
    draw.polygon(points, fill=(248, 248, 242))
    draw.ellipse((x - 4, y - 29, x + 4, y - 21), fill=(248, 248, 242))


def _render(positions: tuple[tuple[int, int], ...]) -> Image.Image:
    image = Image.new("RGB", (512, 512), (25, 34, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 48, 477, 464), fill=(48, 58, 66), outline=(118, 128, 132), width=4)
    draw.rectangle((55, 70, 457, 442), outline=(80, 91, 98), width=2)
    for x in (145, 255, 365):
        draw.line((x, 82, x, 430), fill=(74, 84, 91), width=2)
    for y in (165, 275, 385):
        draw.line((68, y, 444, y), fill=(74, 84, 91), width=2)
    for x, y in positions:
        _draw_aircraft(draw, x, y)
    return image


def main() -> int:
    images_dir = OUTPUT_ROOT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    fixtures = (
        (
            "counterfactual-airport-3",
            "airport_three.png",
            ((145, 165), (255, 275), (365, 385)),
        ),
        (
            "counterfactual-airport-6",
            "airport_six.png",
            (
                (145, 165),
                (255, 165),
                (365, 165),
                (145, 385),
                (255, 385),
                (365, 385),
            ),
        ),
    )

    rows: list[dict[str, object]] = []
    for sample_id, filename, positions in fixtures:
        destination = images_dir / filename
        image = _render(positions)
        image.save(destination, format="PNG", compress_level=9, optimize=False)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        pixel_digest = hashlib.sha256(
            image.mode.encode("ascii")
            + repr(image.size).encode("ascii")
            + image.tobytes()
        ).hexdigest()
        rows.append(
            {
                "sample_id": sample_id,
                "image": f"images/{filename}",
                "question": QUESTION,
                "answer": str(len(positions)),
                "task_type": "synthetic_visual_counting",
                "metadata": {
                    "counterfactual_group": "aircraft-count-v1",
                    "expected_visual_count": len(positions),
                    "fixture_scope": "live_vlm_gate_only",
                    "image_sha256": digest,
                    "pixel_sha256": pixel_digest,
                    "synthetic": True,
                },
            }
        )

    manifest = OUTPUT_ROOT / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"generated {len(rows)} counterfactual fixtures at {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
