"""Artifact descriptors, binary stores, and the Phase 3 latent smoke path."""

from .latent import (
    FLOAT32_LE_MEDIA_TYPE,
    IncompatibleLatentConsumerError,
    LatentConsumer,
    LatentEncoder,
    LatentFeatureConfig,
    LatentFeatureDisabledError,
    LatentReadResult,
    PIXEL_STATS_FEATURE_COUNT,
    PixelStatsLatentEncoder,
)
from .qwen_latent import LatentConsumer as QwenLatentConsumer
from .qwen_native_latent import QwenNativeLatentEncoder
from .qwen_latent import QwenMmprojLatentEncoder
from .store import (
    ArtifactIntegrityError,
    ArtifactKind,
    InMemoryArtifactStore,
    StoredArtifact,
    VisualArtifact,
)

__all__ = [
    "ArtifactIntegrityError",
    "ArtifactKind",
    "FLOAT32_LE_MEDIA_TYPE",
    "InMemoryArtifactStore",
    "IncompatibleLatentConsumerError",
    "LatentConsumer",
    "LatentEncoder",
    "LatentFeatureConfig",
    "LatentFeatureDisabledError",
    "LatentReadResult",
    "PIXEL_STATS_FEATURE_COUNT",
    "PixelStatsLatentEncoder",
    "QwenLatentConsumer",
    "QwenMmprojLatentEncoder",
    "QwenNativeLatentEncoder",
    "StoredArtifact",
    "VisualArtifact",
]
