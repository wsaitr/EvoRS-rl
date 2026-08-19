from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from rs_hmcomm.backends.base import ModelBackend, AgentOutput
from rs_hmcomm.core import SceneTree, MessageBus

@dataclass
class AgentContext:
    image: Any
    question: str
    tree: SceneTree
    bus: MessageBus
    target_node_id: str | None = None

class BaseAgent:
    name: str = "base"

    def __init__(self, backend: ModelBackend):
        self.backend = backend

    def prompt(self, ctx: AgentContext) -> str:
        raise NotImplementedError

    def run(self, ctx: AgentContext) -> AgentOutput:
        return self.backend.generate(ctx.image, self.prompt(ctx), structured_context=ctx.tree.to_dict())
