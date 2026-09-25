"""Calibrate the VLM bbox coordinate canvas with a synthetic probe.

Draws squares at known positions, asks the model for pixel boxes, and prints
the implied canvas so ``_parse_bboxes`` normalization stays honest.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import urllib.request
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vrsbench20_smoke import build_backend  # noqa: E402

from rs_hmcomm.backends.base import BackendRequest, VisualInput  # noqa: E402


PROMPT = (
    "One white square on black background. Reply ONLY JSON: "
    '{{"bbox":[x1,y1,x2,y2]}} using PIXEL coordinates.'
)


def probe(backend: Any, square_pixels: tuple[int, int, int, int], size: int = 512) -> None:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    ImageDraw.Draw(image).rectangle(list(square_pixels), fill=(255, 255, 255))
    request = BackendRequest(
        prompt=PROMPT,
        visuals=(VisualInput(visual_id="probe", label="probe", image=image),),
    )
    output = backend.generate_turn(request)
    text = output.text
    import re

    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if len(numbers) >= 4:
        x1, y1, x2, y2 = numbers[:4]
        gt_norm = tuple(v / size for v in square_pixels)
        print(
            f"square(px)={square_pixels} gt_norm={tuple(round(v,3) for v in gt_norm)}\n"
            f"  model_raw={numbers[:4]}  implied_canvas_x1={x1/gt_norm[0]:.0f}"
        )
    else:
        print(f"square(px)={square_pixels} unparsable: {text[:120]}")


def main() -> int:
    backend = build_backend(Path("configs/models/qwen38_27b_q4_local.yaml"))
    for box in [(51, 51, 153, 153), (307, 205, 410, 358), (256, 256, 461, 461)]:
        probe(backend, box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
