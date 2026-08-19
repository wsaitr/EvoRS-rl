from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import json
import uuid

class NodeLevel(str, Enum):
    SCENE = "scene"
    REGION = "region"
    GROUP = "group"
    OBJECT = "object"

class NodeStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"

class MessageModality(str, Enum):
    TEXT = "text"
    STRUCT = "struct"
    LATENT = "latent"

BBox = tuple[float, float, float, float]

@dataclass
class SceneNode:
    id: str
    level: NodeLevel
    semantic: str
    bbox: Optional[BBox] = None
    confidence: float = 0.0
    status: NodeStatus = NodeStatus.PROPOSED
    parent_id: Optional[str] = None
    evidence_refs: list[str] = field(default_factory=list)
    latent_handle: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(level: NodeLevel, semantic: str, *, parent_id: Optional[str] = None,
             bbox: Optional[BBox] = None, confidence: float = 0.0, **attributes: Any) -> "SceneNode":
        return SceneNode(
            id=f"{level.value}_{uuid.uuid4().hex[:10]}",
            level=level,
            semantic=semantic,
            parent_id=parent_id,
            bbox=bbox,
            confidence=confidence,
            attributes=attributes,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        d["status"] = self.status.value
        return d

class SceneTree:
    def __init__(self, root_semantic: str = "unknown_scene") -> None:
        self.root = SceneNode.make(NodeLevel.SCENE, root_semantic)
        self._nodes: dict[str, SceneNode] = {self.root.id: self.root}
        self._children: dict[str, list[str]] = {self.root.id: []}

    def add_node(self, node: SceneNode) -> SceneNode:
        if node.id in self._nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        if node.parent_id is None:
            node.parent_id = self.root.id
        if node.parent_id not in self._nodes:
            raise KeyError(f"parent does not exist: {node.parent_id}")
        self._nodes[node.id] = node
        self._children.setdefault(node.id, [])
        self._children.setdefault(node.parent_id, []).append(node.id)
        return node

    def get_node(self, node_id: str) -> SceneNode:
        return self._nodes[node_id]

    def children(self, node_id: str) -> list[SceneNode]:
        return [self._nodes[i] for i in self._children.get(node_id, [])]

    def query_by_level(self, level: NodeLevel) -> list[SceneNode]:
        return [n for n in self._nodes.values() if n.level == level]

    def query_by_semantic(self, semantic: str) -> list[SceneNode]:
        s = semantic.lower()
        return [n for n in self._nodes.values() if s in n.semantic.lower()]

    def update_node(self, node_id: str, **changes: Any) -> SceneNode:
        node = self.get_node(node_id)
        for k, v in changes.items():
            if not hasattr(node, k):
                raise AttributeError(k)
            setattr(node, k, v)
        return node

    def verify(self, node_id: str, confidence: Optional[float] = None) -> SceneNode:
        node = self.get_node(node_id)
        node.status = NodeStatus.VERIFIED
        if confidence is not None:
            node.confidence = confidence
        return node

    def reject(self, node_id: str) -> SceneNode:
        node = self.get_node(node_id)
        node.status = NodeStatus.REJECTED
        return node

    def subtree_ids(self, node_id: str) -> list[str]:
        out: list[str] = []
        stack = [node_id]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(reversed(self._children.get(cur, [])))
        return out

    def subtree(self, node_id: str) -> list[SceneNode]:
        return [self._nodes[i] for i in self.subtree_ids(node_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root.id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "children": self._children,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def compact_summary(self) -> str:
        lines: list[str] = []
        def walk(node_id: str, depth: int = 0) -> None:
            n = self._nodes[node_id]
            lines.append(f"{'  '*depth}- {n.level.value}:{n.semantic} [{n.status.value}, conf={n.confidence:.2f}]")
            for c in self._children.get(node_id, []):
                walk(c, depth + 1)
        walk(self.root.id)
        return "\n".join(lines)

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    spatial_level: NodeLevel
    modalities: set[MessageModality] = field(default_factory=lambda: {MessageModality.TEXT})
    text: str = ""
    node_ids: list[str] = field(default_factory=list)
    bbox_refs: list[BBox] = field(default_factory=list)
    latent_handles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

class MessageBus:
    def __init__(self) -> None:
        self.messages: list[AgentMessage] = []

    def send(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def by_receiver(self, receiver: str) -> list[AgentMessage]:
        return [m for m in self.messages if m.receiver == receiver]

    def stats(self) -> dict[str, int]:
        return {
            "messages": len(self.messages),
            "text_chars": sum(len(m.text) for m in self.messages),
            "struct_nodes": sum(len(m.node_ids) for m in self.messages),
            "bbox_refs": sum(len(m.bbox_refs) for m in self.messages),
            "latent_handles": sum(len(m.latent_handles) for m in self.messages),
        }
