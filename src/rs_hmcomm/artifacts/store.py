"""Immutable in-memory artifacts with JSON-safe descriptors.

The descriptor and the binary payload deliberately have different lifetimes:
descriptors may cross Agent/Workspace boundaries, while payload bytes remain in
an ``ArtifactStore`` and are addressed by ``artifact_id``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _non_empty(value: str, field_name: str, *, no_whitespace: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if no_whitespace and any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain whitespace")
    return value


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty(value, field_name)


def _shape(value: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError("shape must be a tuple or list of positive integers")
    normalized = tuple(value)
    if not normalized:
        raise ValueError("shape must not be empty")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in normalized
    ):
        raise ValueError("shape dimensions must be positive integers")
    return normalized


def _roi(
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
        normalized = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise TypeError("roi coordinates must be real numbers") from exc
    if any(not math.isfinite(coordinate) for coordinate in normalized):
        raise ValueError("roi coordinates must be finite")
    x1, y1, x2, y2 = normalized
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            "roi must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )
    return x1, y1, x2, y2


def _consumer_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError("compatible_consumers must be a tuple or list")
    normalized = tuple(
        _non_empty(item, f"compatible_consumers[{index}]", no_whitespace=True)
        for index, item in enumerate(value)
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError("compatible_consumers must not contain duplicates")
    return normalized


class ArtifactKind(str, Enum):
    IMAGE = "image"
    CROP = "crop"
    LATENT = "latent"


@dataclass(frozen=True, slots=True)
class VisualArtifact:
    """Serializable metadata for a binary visual or latent artifact.

    ``source_image_ref`` is an opaque provenance reference.  At a Workspace
    boundary it may resolve either to the task's original image reference or
    to an existing image/crop artifact.  It is never interpreted as a local
    filesystem path by this module.
    """

    artifact_id: str
    kind: ArtifactKind
    media_type: str
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    source_image_ref: str
    roi: tuple[float, float, float, float] | None
    encoder: str | None
    layer: str | None
    projector: str | None
    compatible_consumers: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        _non_empty(self.artifact_id, "artifact_id", no_whitespace=True)
        if not isinstance(self.kind, ArtifactKind):
            try:
                object.__setattr__(self, "kind", ArtifactKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("kind must be image, crop, or latent") from exc
        _non_empty(self.media_type, "media_type")
        object.__setattr__(self, "shape", _shape(self.shape))
        _non_empty(self.dtype, "dtype")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size <= 0
        ):
            raise ValueError("byte_size must be a positive integer")
        _non_empty(self.source_image_ref, "source_image_ref", no_whitespace=True)
        object.__setattr__(self, "roi", _roi(self.roi))
        for field_name in ("encoder", "layer", "projector"):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "compatible_consumers",
            _consumer_ids(self.compatible_consumers),
        )
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if self.kind == ArtifactKind.LATENT:
            if any(
                getattr(self, field_name) is None
                for field_name in ("encoder", "layer", "projector")
            ):
                raise ValueError(
                    "latent artifacts require encoder, layer, and projector metadata"
                )
            if not self.compatible_consumers:
                raise ValueError("latent artifacts require a compatible consumer")

    def descriptor(self) -> dict[str, Any]:
        """Return JSON-safe metadata without any binary payload."""

        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "media_type": self.media_type,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "source_image_ref": self.source_image_ref,
            "roi": list(self.roi) if self.roi is not None else None,
            "encoder": self.encoder,
            "layer": self.layer,
            "projector": self.projector,
            "compatible_consumers": list(self.compatible_consumers),
            "sha256": self.sha256,
        }


class ArtifactIntegrityError(ValueError):
    """Raised when payload bytes do not match their immutable descriptor."""


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    artifact: VisualArtifact
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, VisualArtifact):
            raise TypeError("artifact must be a VisualArtifact")
        if not isinstance(self.data, bytes):
            raise TypeError("stored artifact data must be immutable bytes")

    def descriptor(self) -> dict[str, Any]:
        return self.artifact.descriptor()


class InMemoryArtifactStore:
    """Small immutable store used by Phase 3 smoke and unit tests."""

    def __init__(self) -> None:
        self._items: dict[str, StoredArtifact] = {}

    @staticmethod
    def _validated_data(artifact: VisualArtifact, data: Any) -> bytes:
        if not isinstance(artifact, VisualArtifact):
            raise TypeError("artifact must be a VisualArtifact")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("artifact data must be bytes-like")
        immutable = bytes(data)
        if len(immutable) != artifact.byte_size:
            raise ArtifactIntegrityError(
                "artifact byte_size does not match the supplied payload"
            )
        digest = hashlib.sha256(immutable).hexdigest()
        if digest != artifact.sha256:
            raise ArtifactIntegrityError(
                "artifact sha256 does not match the supplied payload"
            )
        return immutable

    def put(self, artifact: VisualArtifact, data: Any) -> VisualArtifact:
        immutable = self._validated_data(artifact, data)
        if artifact.artifact_id in self._items:
            raise ValueError(f"artifact_id already exists: {artifact.artifact_id}")
        self._items[artifact.artifact_id] = StoredArtifact(artifact, immutable)
        return artifact

    def get(self, artifact_id: str) -> StoredArtifact:
        _non_empty(artifact_id, "artifact_id", no_whitespace=True)
        try:
            stored = self._items[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact_id: {artifact_id}") from exc
        # Revalidate on reads as well as writes so a future non-memory backend
        # cannot silently return corrupt data under a valid descriptor.
        data = self._validated_data(stored.artifact, stored.data)
        return StoredArtifact(stored.artifact, data)

    def descriptor(self, artifact_id: str) -> VisualArtifact:
        return self.get(artifact_id).artifact

    def __contains__(self, artifact_id: object) -> bool:
        return isinstance(artifact_id, str) and artifact_id in self._items

    def __len__(self) -> int:
        return len(self._items)


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactKind",
    "InMemoryArtifactStore",
    "StoredArtifact",
    "VisualArtifact",
]
