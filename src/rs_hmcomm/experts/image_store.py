"""External image registry resolving task image_refs for VLM experts.

Workspace contracts forbid binary payloads inside metadata, so VLM experts
resolve ``task.image_ref`` through this side registry instead.  The registry
is process-local and never serialized.
"""
from __future__ import annotations

from typing import Any


class ImageStore:
    """Process-local mapping from opaque string refs to PIL images."""

    def __init__(self) -> None:
        self._images: dict[str, Any] = {}

    def put(self, image_ref: str, image: Any) -> None:
        if not isinstance(image_ref, str) or not image_ref.strip():
            raise ValueError("image_ref must be a non-empty string")
        if image is None:
            raise ValueError("image must not be None")
        self._images[image_ref] = image

    def get(self, image_ref: str) -> Any | None:
        return self._images.get(image_ref)


__all__ = ["ImageStore"]
