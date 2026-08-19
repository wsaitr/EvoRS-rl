from __future__ import annotations
from dataclasses import dataclass
from rs_hmcomm.core import NodeLevel, SceneTree

@dataclass
class ControllerDecision:
    agent: str
    target_node_id: str | None = None
    stop: bool = False
    reason: str = ""

class RuleController:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def decide(self, question: str, tree: SceneTree, step: int) -> ControllerDecision:
        if step >= self.max_steps:
            return ControllerDecision("verifier", stop=True, reason="budget exhausted")

        regions = tree.query_by_level(NodeLevel.REGION)
        groups = tree.query_by_level(NodeLevel.GROUP)
        objects = tree.query_by_level(NodeLevel.OBJECT)

        if not regions:
            return ControllerDecision("global", reason="need coarse region structure")

        q = question.lower()
        local_needed = any(k in q for k in ["how many", "count", "where", "locate", "aircraft", "ship"])
        if local_needed and not groups and not objects:
            target = max(regions, key=lambda n: n.confidence)
            return ControllerDecision("local", target.id, reason="need localized fine evidence")

        if groups or objects:
            if tree.root.attributes.get("hierarchy_ready"):
                return ControllerDecision("verifier", stop=True, reason="hierarchy assembled; verify evidence")
            return ControllerDecision("hierarchy", reason="aggregate local evidence")

        return ControllerDecision("verifier", stop=True, reason="sufficient for stage-1 rule baseline")
