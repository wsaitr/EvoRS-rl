from __future__ import annotations

from PIL import Image

from rs_hmcomm.backends.base import BackendRequest, VisualInput
from rs_hmcomm.backends.qwen_vl import Qwen25VLBackend


def test_qwen_message_builder_preserves_visual_order_without_loading_weights() -> None:
    backend = object.__new__(Qwen25VLBackend)
    backend.min_pixels = 256 * 28 * 28
    backend.max_pixels = 512 * 28 * 28
    first = Image.new("RGB", (16, 16), "red")
    second = Image.new("RGB", (8, 8), "blue")
    request = BackendRequest(
        prompt="Return JSON",
        visuals=(
            VisualInput("input_image", "input", first),
            VisualInput("visual_crop-1", "crop", second, (0.0, 0.0, 0.5, 0.5)),
        ),
    )

    messages = backend._build_messages(request)
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "image", "text"]
    assert content[0]["image"] is first
    assert content[1]["image"] is second
    assert content[0]["min_pixels"] == backend.min_pixels
    assert content[-1]["text"] == "Return JSON"
