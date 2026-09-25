"""Strict local JSONL manifest adapter for reproducible Phase 1 runs."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from PIL import Image

from .base import RSSample


_REQUIRED_FIELDS = {"sample_id", "image", "question", "answer"}
_OPTIONAL_FIELDS = {"task_type", "bbox", "metadata"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


class ManifestDatasetError(ValueError):
    """Raised when a local dataset manifest violates its contract."""


def _line_error(manifest_path: Path, line_number: int, message: str) -> ManifestDatasetError:
    return ManifestDatasetError(f"{manifest_path}:{line_number}: {message}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_row(line: str, manifest_path: Path, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise _line_error(manifest_path, line_number, "blank lines are not allowed")
    try:
        row = json.loads(line, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _line_error(manifest_path, line_number, f"invalid JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise _line_error(manifest_path, line_number, "each JSONL row must be an object")

    fields = set(row)
    missing = _REQUIRED_FIELDS - fields
    unknown = fields - _ALLOWED_FIELDS
    if missing:
        raise _line_error(manifest_path, line_number, f"missing fields: {sorted(missing)}")
    if unknown:
        raise _line_error(manifest_path, line_number, f"unknown fields: {sorted(unknown)}")
    return row


def _nonempty_string(row: Mapping[str, Any], field_name: str, manifest_path: Path, line_number: int) -> str:
    value = row[field_name]
    if not isinstance(value, str) or not value.strip():
        raise _line_error(manifest_path, line_number, f"{field_name} must be a non-empty string")
    return value


def _optional_task_type(row: Mapping[str, Any], manifest_path: Path, line_number: int) -> str:
    if "task_type" not in row:
        return "vqa"
    value = row["task_type"]
    if not isinstance(value, str) or not value.strip():
        raise _line_error(manifest_path, line_number, "task_type must be a non-empty string")
    return value


def _optional_bbox(
    row: Mapping[str, Any], manifest_path: Path, line_number: int
) -> tuple[float, float, float, float] | None:
    value = row.get("bbox")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise _line_error(manifest_path, line_number, "bbox must contain exactly four numbers")

    coordinates: list[float] = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise _line_error(manifest_path, line_number, f"bbox[{index}] must be a number")
        normalized = float(coordinate)
        if not math.isfinite(normalized):
            raise _line_error(manifest_path, line_number, f"bbox[{index}] must be finite")
        coordinates.append(normalized)
    x1, y1, x2, y2 = coordinates
    if x2 < x1 or y2 < y1:
        raise _line_error(manifest_path, line_number, "bbox must satisfy x2 >= x1 and y2 >= y1")
    return x1, y1, x2, y2


def _optional_metadata(row: Mapping[str, Any], manifest_path: Path, line_number: int) -> dict[str, Any] | None:
    value = row.get("metadata")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _line_error(manifest_path, line_number, "metadata must be a JSON object or null")
    return dict(value)


def _resolve_image(
    image_value: str,
    manifest_root: Path,
    manifest_path: Path,
    line_number: int,
) -> Path:
    relative_path = Path(image_value)
    if relative_path.is_absolute() or relative_path.anchor:
        raise _line_error(manifest_path, line_number, "image must be relative to the manifest directory")
    if ".." in relative_path.parts:
        raise _line_error(manifest_path, line_number, "image path must not contain '..'")

    try:
        resolved = (manifest_root / relative_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _line_error(manifest_path, line_number, f"image cannot be resolved: {exc}") from exc

    try:
        resolved.relative_to(manifest_root)
    except ValueError as exc:
        raise _line_error(
            manifest_path,
            line_number,
            "image resolves outside the manifest directory",
        ) from exc
    if not resolved.is_file():
        raise _line_error(manifest_path, line_number, "image path is not a regular file")
    return resolved


def iter_manifest_dataset(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[RSSample]:
    """Yield samples from a strict local JSONL manifest in file order.

    The image is fully loaded and copied while its source file is open, so the
    yielded ``RSSample`` never retains an open file descriptor.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")

    source = Path(manifest_path).resolve(strict=True)
    if not source.is_file():
        raise ManifestDatasetError(f"manifest is not a regular file: {source}")
    manifest_root = source.parent.resolve(strict=True)
    seen_ids: set[str] = set()
    yielded = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and yielded >= limit:
                break

            row = _load_row(line, source, line_number)
            sample_id = _nonempty_string(row, "sample_id", source, line_number)
            if sample_id in seen_ids:
                raise _line_error(source, line_number, f"duplicate sample_id {sample_id!r}")
            seen_ids.add(sample_id)

            image_value = _nonempty_string(row, "image", source, line_number)
            question = _nonempty_string(row, "question", source, line_number)
            answer = _nonempty_string(row, "answer", source, line_number)
            task_type = _optional_task_type(row, source, line_number)
            bbox = _optional_bbox(row, source, line_number)
            metadata = _optional_metadata(row, source, line_number)
            image_path = _resolve_image(image_value, manifest_root, source, line_number)

            try:
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB").copy()
            except (OSError, ValueError) as exc:
                raise _line_error(source, line_number, f"failed to load image: {exc}") from exc

            yield RSSample(
                sample_id=sample_id,
                image=image,
                question=question,
                answer=answer,
                task_type=task_type,
                bbox=bbox,
                metadata=metadata,
            )
            yielded += 1


__all__ = ["ManifestDatasetError", "iter_manifest_dataset"]
