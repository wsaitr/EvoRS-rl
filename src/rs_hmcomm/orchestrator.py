from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from rs_hmcomm.core import SceneTree, MessageBus
from rs_hmcomm.agents import AgentContext
from rs_hmcomm.controllers import RuleController

@dataclass
class EpisodeResult:
    tree: SceneTree
    bus: MessageBus
    outputs: list[tuple[str, str]] = field(default_factory=list)
    stopped_by: str = ""

class MultiAgentOrchestrator:
    def __init__(self, agents: dict[str, Any], controller: RuleController):
        self.agents = agents
        self.controller = controller

    def run(self, image: Any, question: str) -> EpisodeResult:
        tree = SceneTree()
        bus = MessageBus()
        result = EpisodeResult(tree=tree, bus=bus)

        for step in range(self.controller.max_steps + 1):
            decision = self.controller.decide(question, tree, step)
            agent = self.agents[decision.agent]
            ctx = AgentContext(
                image=image,
                question=question,
                tree=tree,
                bus=bus,
                target_node_id=decision.target_node_id,
            )
            out = agent.run(ctx)
            result.outputs.append((decision.agent, out.text))
            if decision.stop:
                result.stopped_by = decision.reason
                break
        return result
