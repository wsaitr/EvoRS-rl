from __future__ import annotations
from typing import Any
import uuid

class LatentStore:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def put(self, value: Any, prefix: str = "latent") -> str:
        handle = f"{prefix}_{uuid.uuid4().hex[:12]}"
        self._items[handle] = value
        return handle

    def get(self, handle: str) -> Any:
        if handle not in self._items:
            raise KeyError(f"unknown latent handle: {handle}")
        return self._items[handle]

    def pop(self, handle: str) -> Any:
        return self._items.pop(handle)

    def clear(self) -> None:
        self._items.clear()
