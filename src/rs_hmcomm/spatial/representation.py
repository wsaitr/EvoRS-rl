"""Lightweight, JSON-only spatial hierarchy for one remote-sensing image."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from numbers import Real
from typing import Any, Iterable, Mapping

from rs_hmcomm.contracts._validation import (
    coerce_enum,
    json_object_copy,
    normalize_confidence,
    normalize_id_tuple,
    normalize_json_object,
    require_id,
    require_text,
    strict_dict,
)


class SpatialLevel(str, Enum):
    SCENE = "scene"
    REGION = "region"
    GROUP = "group"
    OBJECT = "object"


def _normalize_bbox(
    value: tuple[float, float, float, float] | list[float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError("bbox must be a tuple/list of four normalized numbers or None")
    if len(value) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    coordinates: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"bbox[{index}] must be a real number")
        coordinate = float(item)
        if not math.isfinite(coordinate):
            raise ValueError(f"bbox[{index}] must be finite")
        coordinates.append(coordinate)
    x1, y1, x2, y2 = coordinates
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            "bbox must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class SpatialNode:
    """One question-conditioned spatial description node.

    Images, tensors, embeddings, and binary data stay in an external artifact
    store.  This record carries only stable string references to those values.
    """

    node_id: str
    level: SpatialLevel
    semantic: str
    bbox: tuple[float, float, float, float] | None = None
    parent_id: str | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    evidence_ids: tuple[str, ...] = ()
    visual_artifact_refs: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.node_id, "node_id")
        object.__setattr__(
            self,
            "level",
            coerce_enum(self.level, SpatialLevel, "level"),
        )
        require_text(self.semantic, "semantic")
        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        if self.parent_id is not None:
            require_id(self.parent_id, "parent_id")
            if self.parent_id == self.node_id:
                raise ValueError("a spatial node cannot be its own parent")
        object.__setattr__(
            self,
            "confidence",
            normalize_confidence(self.confidence, "confidence"),
        )
        object.__setattr__(
            self,
            "uncertainty",
            normalize_confidence(self.uncertainty, "uncertainty"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            normalize_id_tuple(self.evidence_ids, "evidence_ids"),
        )
        object.__setattr__(
            self,
            "visual_artifact_refs",
            normalize_id_tuple(
                self.visual_artifact_refs,
                "visual_artifact_refs",
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            normalize_json_object(self.provenance, "provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level.value,
            "semantic": self.semantic,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "parent_id": self.parent_id,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "evidence_ids": list(self.evidence_ids),
            "visual_artifact_refs": list(self.visual_artifact_refs),
            "provenance": json_object_copy(self.provenance),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpatialNode":
        values = strict_dict(
            data,
            cls.__name__,
            required={"node_id", "level", "semantic"},
            optional={
                "bbox",
                "parent_id",
                "confidence",
                "uncertainty",
                "evidence_ids",
                "visual_artifact_refs",
                "provenance",
            },
        )
        return cls(
            node_id=values["node_id"],
            level=values["level"],
            semantic=values["semantic"],
            bbox=values.get("bbox"),
            parent_id=values.get("parent_id"),
            confidence=values.get("confidence"),
            uncertainty=values.get("uncertainty"),
            evidence_ids=values.get("evidence_ids", ()),
            visual_artifact_refs=values.get("visual_artifact_refs", ()),
            provenance=values.get("provenance", {}),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SpatialNode":
        if not isinstance(raw, str):
            raise TypeError("SpatialNode JSON must be a string")

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant is not allowed: {value}")

        data = json.loads(raw, parse_constant=reject_constant)
        if not isinstance(data, Mapping):
            raise ValueError("SpatialNode JSON must contain an object")
        return cls.from_dict(data)


_ALLOWED_PARENT_LEVELS: dict[SpatialLevel, frozenset[SpatialLevel]] = {
    SpatialLevel.SCENE: frozenset(),
    SpatialLevel.REGION: frozenset({SpatialLevel.SCENE}),
    SpatialLevel.GROUP: frozenset({SpatialLevel.SCENE, SpatialLevel.REGION}),
    SpatialLevel.OBJECT: frozenset(
        {SpatialLevel.SCENE, SpatialLevel.REGION, SpatialLevel.GROUP}
    ),
}


def validate_spatial_hierarchy(
    nodes: Iterable[SpatialNode],
) -> tuple[SpatialNode, ...]:
    """Validate a lightweight Scene/Region/Group/Object hierarchy."""

    if isinstance(nodes, (str, bytes)):
        raise TypeError("nodes must be an iterable of SpatialNode records")
    normalized = tuple(nodes)
    if not normalized:
        raise ValueError("spatial hierarchy must not be empty")
    if any(not isinstance(node, SpatialNode) for node in normalized):
        raise TypeError("nodes must contain SpatialNode records only")

    by_id: dict[str, SpatialNode] = {}
    for node in normalized:
        if node.node_id in by_id:
            raise ValueError(f"duplicate spatial node ID: {node.node_id}")
        by_id[node.node_id] = node

    for node in normalized:
        if node.parent_id is not None and node.parent_id not in by_id:
            raise ValueError(
                f"spatial node {node.node_id} has dangling parent {node.parent_id}"
            )

    # Detect cycles independently of level constraints so this stays valid if
    # new hierarchy levels are introduced later.
    state: dict[str, int] = {}

    def visit(node: SpatialNode) -> None:
        marker = state.get(node.node_id, 0)
        if marker == 1:
            raise ValueError("spatial hierarchy contains a parent cycle")
        if marker == 2:
            return
        state[node.node_id] = 1
        if node.parent_id is not None:
            visit(by_id[node.parent_id])
        state[node.node_id] = 2

    for node in normalized:
        visit(node)

    scene_roots = [
        node
        for node in normalized
        if node.level == SpatialLevel.SCENE and node.parent_id is None
    ]
    if len(scene_roots) != 1:
        raise ValueError("spatial hierarchy requires exactly one Scene root")

    for node in normalized:
        if node.parent_id is None:
            if node.level != SpatialLevel.SCENE:
                raise ValueError(
                    f"non-Scene node {node.node_id} must have a parent"
                )
            continue
        parent = by_id[node.parent_id]
        allowed = _ALLOWED_PARENT_LEVELS[node.level]
        if parent.level not in allowed:
            allowed_names = sorted(level.value for level in allowed)
            raise ValueError(
                f"{node.level.value} node {node.node_id} cannot have "
                f"{parent.level.value} parent; allowed={allowed_names}"
            )

    return normalized


__all__ = ["SpatialLevel", "SpatialNode", "validate_spatial_hierarchy"]
