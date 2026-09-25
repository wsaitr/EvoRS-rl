"""Real ROI latent encoder backed by the local Qwen mmproj vision tower.

The llama-server exposes the model's own multimodal projector through its
native ``POST /embedding`` endpoint (``image_data`` field).  Features are the
model's last-token hidden state over an image (+ optional text hint), L2
normalized by the server.  This is a genuine learned representation from the
same backbone that powers the agents, so latent messages stay in-model
consistent — replacing the Phase 3 pixel-statistics smoke encoder.
"""
from __future__ import annotations

import base64
import io
import json
import math
import struct
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from .store import ArtifactKind, InMemoryArtifactStore, VisualArtifact


FLOAT32_LE_MEDIA_TYPE = "application/x-evors-float32-le"
QWEN_MMPROJ_FEATURE_COUNT = 5120


def _normalized_roi(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("roi must be four normalized coordinates")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("roi must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return x1, y1, x2, y2


def _crop(image: Image.Image, roi: tuple[float, float, float, float] | None) -> Image.Image:
    if roi is None:
        return image
    width, height = image.size
    x1, y1, x2, y2 = roi
    bounds = (
        math.floor(x1 * width),
        math.floor(y1 * height),
        math.ceil(x2 * width),
        math.ceil(y2 * height),
    )
    return image.crop(bounds)


class QwenMmprojLatentEncoder:
    """Encode image/ROI features through the running llama-server's mmproj.

    The endpoint returns the last-token hidden state (pooling=last), which is
    a dense semantic summary of the visual content; the server L2-normalizes.
    """

    encoder_id = "qwen_mmproj_v1"
    layer = "mmproj_last_hidden_l2"
    projector = "qwen38_27b_mmproj_q8"

    def __init__(
        self,
        store: InMemoryArtifactStore,
        *,
        base_url: str = "http://127.0.0.1:18080",
        model: str = "evors/qwen3.8-27b-q4",
        timeout: float = 120.0,
        max_side: int = 448,
        expected_dim: int = QWEN_MMPROJ_FEATURE_COUNT,
    ) -> None:
        self.store = store
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_side = max_side
        self.expected_dim = expected_dim

    # -- transport ---------------------------------------------------------

    def _post_embedding(self, payload: dict[str, Any]) -> list[float]:
        request = urllib.request.Request(
            f"{self.base_url}/embedding",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"llama-server /embedding failed: {detail}") from exc
        items = body if isinstance(body, list) else [body]
        embedding = items[0]["embedding"]
        vector = embedding[0] if isinstance(embedding[0], list) else embedding
        if len(vector) != self.expected_dim:
            raise RuntimeError(
                f"unexpected embedding dimension {len(vector)} "
                f"(expected {self.expected_dim})"
            )
        return [float(value) for value in vector]

    @staticmethod
    def _to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # -- LatentEncoder protocol --------------------------------------------

    def encode(
        self,
        image: Image.Image,
        *,
        artifact_id: str,
        source_image_ref: str,
        roi: tuple[float, float, float, float] | None = None,
        compatible_consumers: tuple[str, ...] = ("reasoning",),
        text_hint: str | None = None,
    ) -> VisualArtifact:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        normalized_roi = _normalized_roi(roi)
        region = _crop(image, normalized_roi).convert("RGB")
        if region.width <= 0 or region.height <= 0:
            raise ValueError("roi produces an empty image region")

        # Downscale so vision preprocessing is fast and stable.
        scale = self.max_side / max(region.width, region.height)
        if scale < 1.0:
            region = region.resize((max(28, round(region.width * scale)),
                                    max(28, round(region.height * scale))))
        encoded_image = self._to_base64(region)

        prompt = text_hint or "Describe the remote-sensing content of this image."
        payload: dict[str, Any] = {
            "input": prompt,
            "image_data": [{"data": encoded_image}],
        }
        vector = self._post_embedding(payload)

        data = struct.pack(f"<{len(vector)}f", *vector)
        artifact = VisualArtifact(
            artifact_id=artifact_id,
            kind=ArtifactKind.LATENT,
            media_type=FLOAT32_LE_MEDIA_TYPE,
            shape=(len(vector),),
            dtype="float32",
            byte_size=len(data),
            source_image_ref=source_image_ref,
            roi=normalized_roi,
            encoder=self.encoder_id,
            layer=self.layer,
            projector=self.projector,
            compatible_consumers=tuple(compatible_consumers),
            sha256=__import__("hashlib").sha256(data).hexdigest(),
        )
        return self.store.put(artifact, data)


class LatentConsumer:
    """Load only latent artifacts explicitly compatible with it."""

    def __init__(
        self,
        store: InMemoryArtifactStore,
        consumer_id: str,
    ) -> None:
        self.store = store
        self.consumer_id = consumer_id

    def consume(self, artifact_id: str) -> tuple[float, ...]:
        stored = self.store.get(artifact_id)
        artifact = stored.artifact
        if artifact.kind != ArtifactKind.LATENT:
            raise ValueError("artifact is not a latent")
        if self.consumer_id not in artifact.compatible_consumers:
            raise PermissionError(
                f"consumer {self.consumer_id!r} is not compatible with {artifact_id!r}"
            )
        if artifact.media_type != FLOAT32_LE_MEDIA_TYPE or artifact.dtype != "float32":
            raise ValueError("latent encoding is not supported by this consumer")
        element_count = math.prod(artifact.shape)
        expected_bytes = element_count * 4
        if expected_bytes != artifact.byte_size or expected_bytes != len(stored.data):
            raise ValueError("latent shape and byte_size are inconsistent")
        return struct.unpack(f"<{element_count}f", stored.data)


__all__ = [
    "FLOAT32_LE_MEDIA_TYPE",
    "LatentConsumer",
    "QWEN_MMPROJ_FEATURE_COUNT",
    "QwenMmprojLatentEncoder",
]
