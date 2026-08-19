from __future__ import annotations
from rs_hmcomm.core import AgentMessage, MessageModality, NodeLevel, SceneNode
from .base import BaseAgent, AgentContext

class GlobalAgent(BaseAgent):
    name = "global"

    def prompt(self, ctx: AgentContext) -> str:
        return (
            "You are the GLOBAL remote-sensing agent. Analyze the whole image and question. "
            "Propose only high-value functional regions. Return concise grounded evidence. "
            f"Question: {ctx.question}"
        )

    def run(self, ctx: AgentContext):
        out = super().run(ctx)
        s = out.structured
        if s.get("level") == "region" and s.get("bbox"):
            node = SceneNode.make(
                NodeLevel.REGION,
                s.get("semantic", "candidate_region"),
                bbox=tuple(s["bbox"]),
                confidence=out.confidence,
            )
            ctx.tree.add_node(node)
            ctx.bus.send(AgentMessage(
                sender=self.name,
                receiver="local",
                spatial_level=NodeLevel.REGION,
                modalities={MessageModality.TEXT, MessageModality.STRUCT},
                text=out.text,
                node_ids=[node.id],
                bbox_refs=[node.bbox] if node.bbox else [],
            ))
        return out

class LocalAgent(BaseAgent):
    name = "local"

    def prompt(self, ctx: AgentContext) -> str:
        target = ctx.target_node_id or "unspecified"
        return (
            "You are the LOCAL remote-sensing agent. Inspect the target region for fine details, "
            "small objects, groups, counts or localized evidence. "
            f"Target node: {target}. Question: {ctx.question}"
        )

    def run(self, ctx: AgentContext):
        out = super().run(ctx)
        s = out.structured
        parent = ctx.target_node_id or ctx.tree.root.id
        if s.get("level") == "group" and s.get("bbox"):
            node = SceneNode.make(
                NodeLevel.GROUP,
                s.get("semantic", "candidate_group"),
                parent_id=parent,
                bbox=tuple(s["bbox"]),
                confidence=out.confidence,
            )
            ctx.tree.add_node(node)
            ctx.bus.send(AgentMessage(
                sender=self.name,
                receiver="hierarchy",
                spatial_level=NodeLevel.GROUP,
                modalities={MessageModality.TEXT, MessageModality.STRUCT},
                text=out.text,
                node_ids=[node.id],
                bbox_refs=[node.bbox] if node.bbox else [],
            ))
        return out

class HierarchyAgent(BaseAgent):
    name = "hierarchy"

    def prompt(self, ctx: AgentContext) -> str:
        return (
            "You are the HIERARCHY agent. Use the shared Scene->Region->Group->Object tree. "
            "Infer part-of, inside, adjacency and grouping relations without inventing objects. "
            f"Question: {ctx.question}"
        )

    def run(self, ctx: AgentContext):
        out = super().run(ctx)
        ctx.tree.root.attributes["hierarchy_ready"] = True
        ctx.bus.send(AgentMessage(
            sender=self.name,
            receiver="verifier",
            spatial_level=NodeLevel.SCENE,
            modalities={MessageModality.TEXT, MessageModality.STRUCT},
            text=out.text,
            node_ids=ctx.tree.subtree_ids(ctx.tree.root.id),
        ))
        return out

class VerifierAgent(BaseAgent):
    name = "verifier"

    def prompt(self, ctx: AgentContext) -> str:
        return (
            "You are the VERIFIER. Check whether current evidence is sufficient, grounded and "
            "non-contradictory. Mark weak nodes for repair rather than guessing. "
            f"Question: {ctx.question}"
        )

class ResidualAgent(BaseAgent):
    name = "residual"

    def prompt(self, ctx: AgentContext) -> str:
        return (
            "You are an adaptive residual agent. Find information gaps not already covered by "
            "the team. Avoid repeating existing evidence. "
            f"Question: {ctx.question}"
        )
