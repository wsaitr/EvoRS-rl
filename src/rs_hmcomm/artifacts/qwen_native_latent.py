"""Qwen-native ROI latent encoder using the model's own vision tower.

Replaces the broken ``QwenMmprojLatentEncoder`` (which, due to a llama-server
limitation, was returning the text prompt vector regardless of image content).
This encoder loads Qwen3.8's vision tower directly via transformers and runs
a real forward pass on the image/ROI pixels.  The output is a 5120-dim
learned visual feature in the **same representation space** as the VLM agent,
matching the paper's "shared backbone" design principle.

Performance: ~9s/image on CPU (0.4B vision tower, bf16).  The model is
instantiated lazily on first use and cached for the process lifetime.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .store import ArtifactKind, InMemoryArtifactStore, VisualArtifact


FLOAT32_LE_MEDIA_TYPE = "application/x-evors-float32-le"
QWEN_NATIVE_FEATURE_COUNT = 5120

# Default local model layout; overridable per instance.
DEFAULT_CONFIG_PATH = Path(".models/qwen3.8-vision-tower/config.json")
DEFAULT_SHARD_PATH = Path(".models/qwen3.8-vision-tower/model-00001-of-00018.safetensors")
DEFAULT_WEIGHT_PREFIX = "model.visual."


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
    return image.crop((
        math.floor(x1 * width),
        math.floor(y1 * height),
        math.ceil(x2 * width),
        math.ceil(y2 * height),
    ))


class _VisionTower:
    """Lazy-loaded Qwen3.5 vision tower + image processor."""

    _instance: "_VisionTower | None" = None

    def __init__(self, config_path: Path, shard_path: Path, weight_prefix: str):
        from transformers import (
            AutoImageProcessor,
            Qwen3_5VisionConfig,
            Qwen3_5VisionModel,
        )
        from safetensors.torch import load_file

        with config_path.open(encoding="utf-8") as handle:
            full_cfg = json.load(handle)
        vcfg = Qwen3_5VisionConfig(**full_cfg["vision_config"])
        self.model = (
            Qwen3_5VisionModel(vcfg)
            .to(dtype=torch.bfloat16, device="cpu")
            .eval()
        )
        vis_sd = {}
        for key, tensor in load_file(shard_path).items():
            if key.startswith(weight_prefix):
                vis_sd[key[len(weight_prefix):]] = tensor
        missing, unexpected = self.model.load_state_dict(vis_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"vision tower weights mismatch: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        self.processor = AutoImageProcessor.from_pretrained(
            "Qwen/Qwen3.5-27B",
            trust_remote_code=True,
        )

    @classmethod
    def get(
        cls,
        config_path: Path = DEFAULT_CONFIG_PATH,
        shard_path: Path = DEFAULT_SHARD_PATH,
        weight_prefix: str = DEFAULT_WEIGHT_PREFIX,
    ) -> "_VisionTower":
        if cls._instance is None:
            cls._instance = cls(config_path, shard_path, weight_prefix)
        return cls._instance

    @torch.no_grad()
    def encode_region(self, image: Image.Image) -> tuple[float, ...]:
        """Encode one PIL image/ROI to a L2-normalized 5120-d vector."""
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.model(
            hidden_states=inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"],
        )
        pooled = out.pooler_output.float().mean(dim=0).cpu().numpy()
        norm = math.sqrt(float((pooled @ pooled)))
        if norm <= 0:
            raise RuntimeError("vision tower produced a zero vector")
        return tuple(float(value) / norm for value in pooled)


class QwenNativeLatentEncoder:
    """Encode image/ROI features with Qwen3.8's own vision tower.

    Output: 5120-dim float32 vector, L2-normalized, produced by mean-pooling
    over the 256 spatial tokens emitted by the vision merger.  Each vector
    reflects the actual pixel content of the image/ROI (verified to differ
    across images and ROIs in tests/test_qwen_native_latent.py).
    """

    encoder_id = "qwen_native_v1"
    layer = "qwen35_vision_pooler"
    projector = "mean_over_spatial_tokens"

    def __init__(
        self,
        store: InMemoryArtifactStore,
        *,
        config_path: Path = DEFAULT_CONFIG_PATH,
        shard_path: Path = DEFAULT_SHARD_PATH,
        expected_dim: int = QWEN_NATIVE_FEATURE_COUNT,
        tower: _VisionTower | None = None,
    ) -> None:
        self.store = store
        self.expected_dim = expected_dim
        self.tower = tower or _VisionTower.get(config_path, shard_path, DEFAULT_WEIGHT_PREFIX)

    def encode(
        self,
        image: Image.Image,
        *,
        artifact_id: str,
        source_image_ref: str,
        roi: tuple[float, float, float, float] | None = None,
        compatible_consumers: tuple[str, ...] = ("reasoning",),
        text_hint: str | None = None,  # accepted for API symmetry; unused
    ) -> VisualArtifact:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        normalized_roi = _normalized_roi(roi)
        region = _crop(image, normalized_roi).convert("RGB")
        if region.width <= 0 or region.height <= 0:
            raise ValueError("roi produces an empty image region")

        vector = self.tower.encode_region(region)
        if len(vector) != self.expected_dim:
            raise RuntimeError(
                f"unexpected feature dimension {len(vector)} "
                f"(expected {self.expected_dim})"
            )
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
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return self.store.put(artifact, data)


__all__ = [
    "FLOAT32_LE_MEDIA_TYPE",
    "QWEN_NATIVE_FEATURE_COUNT",
    "QwenNativeLatentEncoder",
]
