"""Dataset identity and deterministic sampling contracts for experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from .runtime import validate_json_value
from ._validation import strict_dict


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    revision: str
    split: str
    seed: int
    limit: int | None = None
    manifest_digest: str | None = None

    def __post_init__(self) -> None:
        _required(self.name, "name")
        _required(self.revision, "revision")
        _required(self.split, "split")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
                raise ValueError("limit must be a positive integer or None")
        if self.manifest_digest is not None:
            digest = self.manifest_digest.casefold()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("manifest_digest must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "split": self.split,
            "seed": self.seed,
            "limit": self.limit,
            "manifest_digest": self.manifest_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetSpec":
        values = strict_dict(
            data,
            cls.__name__,
            required={"name", "revision", "split", "seed", "limit", "manifest_digest"},
        )
        return cls(
            name=values["name"],
            revision=values["revision"],
            split=values["split"],
            seed=values["seed"],
            limit=values["limit"],
            manifest_digest=values["manifest_digest"],
        )


def canonical_manifest_digest(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash manifest content using canonical JSON, including record order."""

    normalized = [validate_json_value(dict(record), "manifest record") for record in records]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_sample_ids(
    sample_ids: Iterable[str],
    *,
    seed: int,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Select IDs reproducibly, independent of Python RNG implementation."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    ids = tuple(sample_ids)
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("sample IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("sample IDs must be unique")

    def rank(sample_id: str) -> tuple[bytes, str]:
        payload = f"{seed}\0{sample_id}".encode("utf-8")
        return hashlib.sha256(payload).digest(), sample_id

    ordered = tuple(sorted(ids, key=rank))
    return ordered if limit is None else ordered[:limit]
