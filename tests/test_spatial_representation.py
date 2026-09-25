from __future__ import annotations

import json
import math

import pytest
from PIL import Image

from rs_hmcomm.spatial import (
    SpatialLevel,
    SpatialNode,
    validate_spatial_hierarchy,
)


def scene(node_id: str = "scene-1") -> SpatialNode:
    return SpatialNode(node_id, SpatialLevel.SCENE, "airport scene")


def test_spatial_node_json_roundtrip_contains_refs_but_no_heavy_objects() -> None:
    node = SpatialNode(
        node_id="object-aircraft-1",
        level="object",
        semantic="aircraft",
        bbox=(0.1, 0.2, 0.4, 0.6),
        parent_id="region-apron-1",
        confidence=0.9,
        uncertainty=0.1,
        evidence_ids=("evidence-detection-1",),
        visual_artifact_refs=("artifact-roi-1",),
        provenance={"tool": "detector", "params": {"scale": 2}},
    )

    encoded = node.to_json()
    decoded = SpatialNode.from_json(encoded)

    assert decoded == node
    assert decoded.level is SpatialLevel.OBJECT
    assert decoded.bbox == (0.1, 0.2, 0.4, 0.6)
    assert json.loads(encoded)["visual_artifact_refs"] == ["artifact-roi-1"]
    assert "Image" not in encoded
    assert "tensor" not in encoded.lower()


def test_spatial_node_rejects_heavy_provenance_objects() -> None:
    with pytest.raises(TypeError, match="store images, tensors and binary data externally"):
        SpatialNode(
            "object-1",
            SpatialLevel.OBJECT,
            "aircraft",
            provenance={"image": Image.new("RGB", (2, 2))},
        )


def test_valid_hierarchy_allows_lightweight_direct_parent_shortcuts() -> None:
    nodes = (
        scene(),
        SpatialNode(
            "region-1",
            SpatialLevel.REGION,
            "apron",
            parent_id="scene-1",
        ),
        SpatialNode(
            "group-region",
            SpatialLevel.GROUP,
            "aircraft formation",
            parent_id="region-1",
        ),
        SpatialNode(
            "group-scene",
            SpatialLevel.GROUP,
            "runway system",
            parent_id="scene-1",
        ),
        SpatialNode(
            "object-group",
            SpatialLevel.OBJECT,
            "aircraft",
            parent_id="group-region",
        ),
        SpatialNode(
            "object-region",
            SpatialLevel.OBJECT,
            "hangar",
            parent_id="region-1",
        ),
        SpatialNode(
            "object-scene",
            SpatialLevel.OBJECT,
            "runway",
            parent_id="scene-1",
        ),
    )

    assert validate_spatial_hierarchy(nodes) == nodes


@pytest.mark.parametrize(
    "nodes",
    [
        (SpatialNode("region-1", "region", "apron"),),
        (scene("scene-1"), scene("scene-2")),
    ],
)
def test_hierarchy_requires_exactly_one_scene_root(
    nodes: tuple[SpatialNode, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly one Scene root|must have a parent"):
        validate_spatial_hierarchy(nodes)


@pytest.mark.parametrize(
    ("child_level", "parent_level"),
    [
        (SpatialLevel.REGION, SpatialLevel.REGION),
        (SpatialLevel.GROUP, SpatialLevel.GROUP),
        (SpatialLevel.OBJECT, SpatialLevel.OBJECT),
    ],
)
def test_hierarchy_rejects_illegal_parent_levels(
    child_level: SpatialLevel,
    parent_level: SpatialLevel,
) -> None:
    parent = SpatialNode(
        "parent",
        parent_level,
        "parent",
        parent_id="scene-1",
    )
    child = SpatialNode("child", child_level, "child", parent_id="parent")
    with pytest.raises(ValueError, match="cannot have"):
        validate_spatial_hierarchy((scene(), parent, child))


def test_hierarchy_rejects_duplicate_and_dangling_ids() -> None:
    with pytest.raises(ValueError, match="duplicate spatial node ID"):
        validate_spatial_hierarchy((scene(), scene()))

    with pytest.raises(ValueError, match="dangling parent"):
        validate_spatial_hierarchy(
            (
                scene(),
                SpatialNode(
                    "object-1",
                    SpatialLevel.OBJECT,
                    "aircraft",
                    parent_id="missing-region",
                ),
            )
        )


def test_spatial_node_rejects_self_parent_and_hierarchy_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="own parent"):
        SpatialNode(
            "object-1",
            SpatialLevel.OBJECT,
            "aircraft",
            parent_id="object-1",
        )

    cycle = (
        scene(),
        SpatialNode("group-a", SpatialLevel.GROUP, "a", parent_id="group-b"),
        SpatialNode("group-b", SpatialLevel.GROUP, "b", parent_id="group-a"),
    )
    with pytest.raises(ValueError, match="parent cycle"):
        validate_spatial_hierarchy(cycle)


@pytest.mark.parametrize(
    "bbox",
    [
        (0.1, 0.2, 0.3),
        (0.5, 0.1, 0.5, 0.9),
        (-0.1, 0.1, 0.5, 0.5),
        (0.1, 0.1, 1.1, 0.5),
        (0.1, 0.1, math.inf, 0.5),
        (0.1, 0.1, True, 0.5),
    ],
)
def test_spatial_node_rejects_invalid_bbox(bbox: tuple[float, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="bbox"):
        SpatialNode("object-1", SpatialLevel.OBJECT, "aircraft", bbox=bbox)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", math.nan),
        ("uncertainty", -0.1),
        ("uncertainty", 1.1),
        ("uncertainty", math.inf),
    ],
)
def test_spatial_node_rejects_invalid_confidence_or_uncertainty(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        SpatialNode(
            "object-1",
            SpatialLevel.OBJECT,
            "aircraft",
            **{field_name: value},
        )
