"""Integration test for QwenNativeLatentEncoder: verifies vectors are
content-sensitive (different images/ROIs produce different 5120-d vectors).

Skips if the vision-tower shard is not present locally.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from PIL import Image

from rs_hmcomm.artifacts import InMemoryArtifactStore

# Skip when vision-tower shard is not on disk (e.g. CI).
_SHARD_PATH = Path(".models/qwen3.8-vision-tower/model-00001-of-00018.safetensors")
requires_vision_shard = pytest.mark.skipif(
    not _SHARD_PATH.exists(),
    reason=f"vision-tower shard not present at {_SHARD_PATH}",
)


def _cos(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def _consume(store: InMemoryArtifactStore, artifact_id: str) -> tuple[float, ...]:
    import struct

    stored = store.get(artifact_id)
    n = stored.artifact.shape[0]
    return struct.unpack(f"<{n}f", stored.data)


@requires_vision_shard
def test_different_images_yield_different_vectors():
    from rs_hmcomm.artifacts import QwenNativeLatentEncoder

    store = InMemoryArtifactStore()
    enc = QwenNativeLatentEncoder(store)

    img_a = Image.open("data/vrsbench/val/images/P1752_0051.png").convert("RGB")
    img_b = Image.open("data/vrsbench/val/images/05908_0000.png").convert("RGB")

    enc.encode(
        img_a,
        artifact_id="a",
        source_image_ref="x",
        compatible_consumers=("reasoning",),
    )
    enc.encode(
        img_b,
        artifact_id="b",
        source_image_ref="x",
        compatible_consumers=("reasoning",),
    )

    v_a = _consume(store, "a")
    v_b = _consume(store, "b")

    assert len(v_a) == 5120
    assert len(v_b) == 5120
    # Content-sensitive: different images must yield meaningfully different vectors.
    cosine = _cos(v_a, v_b)
    assert cosine < 0.999, f"vectors appear identical across images (cos={cosine:.4f})"
    # But not wildly unrelated (remote sensing images share scene-level statistics).
    assert cosine > 0.90, f"vectors too dissimilar for similar-domain images (cos={cosine:.4f})"


@requires_vision_shard
def test_repeated_encode_is_deterministic():
    from rs_hmcomm.artifacts import QwenNativeLatentEncoder

    store = InMemoryArtifactStore()
    enc = QwenNativeLatentEncoder(store)

    img = Image.open("data/vrsbench/val/images/P1752_0051.png").convert("RGB")
    enc.encode(
        img, artifact_id="r1", source_image_ref="x", compatible_consumers=("reasoning",)
    )
    enc.encode(
        img, artifact_id="r2", source_image_ref="x", compatible_consumers=("reasoning",)
    )

    v1 = _consume(store, "r1")
    v2 = _consume(store, "r2")
    # Bit-exact determinism expected (no stochastic dropout at inference).
    assert v1 == v2, "encoder is not deterministic across calls"


@requires_vision_shard
def test_roi_crops_produce_different_vectors():
    from rs_hmcomm.artifacts import QwenNativeLatentEncoder

    store = InMemoryArtifactStore()
    enc = QwenNativeLatentEncoder(store)

    img = Image.open("data/vrsbench/val/images/P1752_0051.png").convert("RGB")
    enc.encode(
        img,
        artifact_id="full",
        source_image_ref="x",
        roi=None,
        compatible_consumers=("reasoning",),
    )
    enc.encode(
        img,
        artifact_id="tl",
        source_image_ref="x",
        roi=(0.0, 0.0, 0.5, 0.5),
        compatible_consumers=("reasoning",),
    )
    enc.encode(
        img,
        artifact_id="br",
        source_image_ref="x",
        roi=(0.5, 0.5, 1.0, 1.0),
        compatible_consumers=("reasoning",),
    )

    v_full = _consume(store, "full")
    v_tl = _consume(store, "tl")
    v_br = _consume(store, "br")
    # ROI crops must differ from full image and from each other.
    assert _cos(v_full, v_tl) < 0.999
    assert _cos(v_full, v_br) < 0.999
    assert _cos(v_tl, v_br) < 0.999
