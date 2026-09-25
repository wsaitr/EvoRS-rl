"""A deterministic pixel-statistics latent smoke path.

``PixelStatsLatentEncoder`` proves the ArtifactStore/latent communication
mechanism using real image pixels.  It is intentionally *not* a learned vision
representation and must not be reported as the paper's final latent encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Protocol

from PIL import Image, ImageStat

from .store import ArtifactKind, InMemoryArtifactStore, VisualArtifact


FLOAT32_LE_MEDIA_TYPE = "application/x-evors-float32-le"
PIXEL_STATS_FEATURE_COUNT = 12


@dataclass(frozen=True, slots=True)
class LatentFeatureConfig:
    """Feature flag for the Phase 3 latent path; opt-in by design."""

    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")


class LatentFeatureDisabledError(RuntimeError):
    pass


class IncompatibleLatentConsumerError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class LatentReadResult:
    artifact_id: str
    values: tuple[float, ...]
    bytes_loaded: int


class LatentEncoder(Protocol):
    encoder_id: str
    layer: str
    projector: str

    def encode(
        self,
        image: Image.Image,
        *,
        artifact_id: str,
        source_image_ref: str,
        roi: tuple[float, float, float, float] | None,
        compatible_consumers: tuple[str, ...],
    ) -> VisualArtifact:
        ...


def _normalized_roi(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError("roi must be four normalized coordinates or None")
    if len(value) != 4:
        raise ValueError("roi must contain four normalized coordinates")
    if any(isinstance(coordinate, bool) for coordinate in value):
        raise TypeError("roi coordinates must be real numbers")
    try:
        roi = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise TypeError("roi coordinates must be real numbers") from exc
    if any(not math.isfinite(coordinate) for coordinate in roi):
        raise ValueError("roi coordinates must be finite")
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            "roi must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )
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


class PixelStatsLatentEncoder:
    """Encode normalized RGB mean/std/min/max as twelve float32 values."""

    encoder_id = "pixel_stats_smoke_v1"
    layer = "rgb_mean_std_min_max"
    projector = "identity_float32"

    def __init__(
        self,
        store: InMemoryArtifactStore,
        config: LatentFeatureConfig | None = None,
    ) -> None:
        if not isinstance(store, InMemoryArtifactStore):
            raise TypeError("store must be an InMemoryArtifactStore")
        if config is not None and not isinstance(config, LatentFeatureConfig):
            raise TypeError("config must be a LatentFeatureConfig or None")
        self.store = store
        self.config = config or LatentFeatureConfig()

    def encode(
        self,
        image: Image.Image,
        *,
        artifact_id: str,
        source_image_ref: str,
        roi: tuple[float, float, float, float] | None = None,
        compatible_consumers: tuple[str, ...] = ("reasoning",),
    ) -> VisualArtifact:
        if not self.config.enabled:
            raise LatentFeatureDisabledError("latent feature is disabled")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("image dimensions must be positive")
        normalized_roi = _normalized_roi(roi)
        region = _crop(image, normalized_roi).convert("RGB")
        if region.width <= 0 or region.height <= 0:
            raise ValueError("roi produces an empty image region")

        statistics = ImageStat.Stat(region)
        extrema = region.getextrema()
        means = tuple(float(value) / 255.0 for value in statistics.mean)
        standard_deviations = tuple(float(value) / 255.0 for value in statistics.stddev)
        minima = tuple(float(bounds[0]) / 255.0 for bounds in extrema)
        maxima = tuple(float(bounds[1]) / 255.0 for bounds in extrema)
        features = means + standard_deviations + minima + maxima
        if len(features) != PIXEL_STATS_FEATURE_COUNT:  # pragma: no cover - RGB invariant
            raise RuntimeError("pixel statistics produced an unexpected feature shape")
        data = struct.pack(f"<{len(features)}f", *features)
        artifact = VisualArtifact(
            artifact_id=artifact_id,
            kind=ArtifactKind.LATENT,
            media_type=FLOAT32_LE_MEDIA_TYPE,
            shape=(len(features),),
            dtype="float32",
            byte_size=len(data),
            source_image_ref=source_image_ref,
            roi=normalized_roi,
            encoder=self.encoder_id,
            layer=self.layer,
            projector=self.projector,
            compatible_consumers=compatible_consumers,
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return self.store.put(artifact, data)


class LatentConsumer:
    """Load and parse only latent artifacts explicitly compatible with it."""

    def __init__(
        self,
        store: InMemoryArtifactStore,
        consumer_id: str,
        config: LatentFeatureConfig | None = None,
    ) -> None:
        if not isinstance(store, InMemoryArtifactStore):
            raise TypeError("store must be an InMemoryArtifactStore")
        if not isinstance(consumer_id, str) or not consumer_id.strip():
            raise ValueError("consumer_id must be a non-empty string")
        if any(character.isspace() for character in consumer_id):
            raise ValueError("consumer_id must not contain whitespace")
        if config is not None and not isinstance(config, LatentFeatureConfig):
            raise TypeError("config must be a LatentFeatureConfig or None")
        self.store = store
        self.consumer_id = consumer_id
        self.config = config or LatentFeatureConfig()

    def consume(self, artifact_id: str) -> LatentReadResult:
        if not self.config.enabled:
            raise LatentFeatureDisabledError("latent feature is disabled")
        stored = self.store.get(artifact_id)
        artifact = stored.artifact
        if artifact.kind != ArtifactKind.LATENT:
            raise ValueError("artifact is not a latent")
        if self.consumer_id not in artifact.compatible_consumers:
            raise IncompatibleLatentConsumerError(
                f"consumer {self.consumer_id!r} is not compatible with {artifact_id!r}"
            )
        if artifact.media_type != FLOAT32_LE_MEDIA_TYPE or artifact.dtype != "float32":
            raise ValueError("latent encoding is not supported by this consumer")
        element_count = math.prod(artifact.shape)
        expected_bytes = element_count * 4
        if expected_bytes != artifact.byte_size or expected_bytes != len(stored.data):
            raise ValueError("latent shape and byte_size are inconsistent")
        values = struct.unpack(f"<{element_count}f", stored.data)
        return LatentReadResult(
            artifact_id=artifact.artifact_id,
            values=tuple(float(value) for value in values),
            bytes_loaded=len(stored.data),
        )


__all__ = [
    "FLOAT32_LE_MEDIA_TYPE",
    "IncompatibleLatentConsumerError",
    "LatentConsumer",
    "LatentEncoder",
    "LatentFeatureConfig",
    "LatentFeatureDisabledError",
    "LatentReadResult",
    "PIXEL_STATS_FEATURE_COUNT",
    "PixelStatsLatentEncoder",
]
