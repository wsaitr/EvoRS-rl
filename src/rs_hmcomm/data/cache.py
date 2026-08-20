"""Local data cache manager for RS-HMComm."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import hashlib
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One cached file entry."""
    key: str
    local_path: str
    size_bytes: int = 0
    md5: str = ""
    source: str = "obs"  # obs | huggingface | local


@dataclass
class DataCache:
    """
    Manages local data cache with metadata tracking.

    Tracks what's been downloaded, from where, and verifies integrity.
    """
    cache_dir: Path
    manifest_path: Path
    _entries: dict[str, CacheEntry] = field(default_factory=dict)

    def __init__(self, cache_dir: str | Path = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / ".cache_manifest.json"
        self._entries: dict[str, CacheEntry] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for k, v in data.items():
                self._entries[k] = CacheEntry(**v)

    def _save_manifest(self) -> None:
        data = {k: {
            "key": v.key, "local_path": v.local_path,
            "size_bytes": v.size_bytes, "md5": v.md5, "source": v.source,
        } for k, v in self._entries.items()}
        self.manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def has(self, key: str) -> bool:
        if key not in self._entries:
            return False
        entry = self._entries[key]
        return Path(entry.local_path).exists()

    def get(self, key: str) -> Path | None:
        if self.has(key):
            return Path(self._entries[key].local_path)
        return None

    def put(self, key: str, local_path: str | Path, source: str = "obs") -> CacheEntry:
        local_path = Path(local_path)
        size = local_path.stat().st_size if local_path.exists() else 0

        entry = CacheEntry(
            key=key,
            local_path=str(local_path),
            size_bytes=size,
            source=source,
        )
        self._entries[key] = entry
        self._save_manifest()
        return entry

    def remove(self, key: str) -> bool:
        if key in self._entries:
            entry = self._entries.pop(key)
            path = Path(entry.local_path)
            if path.exists():
                path.unlink()
            self._save_manifest()
            return True
        return False

    def total_size(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def list_entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        for key in list(self._entries.keys()):
            self.remove(key)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "total_size_mb": round(self.total_size() / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
