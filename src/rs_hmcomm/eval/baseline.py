"""Baseline methods for comparison experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from rs_hmcomm.core import SceneTree, MessageBus
from rs_hmcomm.backends.base import AgentOutput, ModelBackend
from rs_hmcomm.orchestrator import EpisodeResult
from rs_hmcomm.agents.base import AgentContext
from rs_hmcomm.agents.roles import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
from rs_hmcomm.controllers.rule import RuleController
from rs_hmcomm.orchestrator import MultiAgentOrchestrator


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    method: str
    answer: str = ""
    task_score: float = 0.0
    communication_cost: float = 0.0
    text_tokens: int = 0
    struct_nodes: int = 0
    latent_bytes: int = 0
    image_crops: int = 0
    agent_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaselineRunner:
    """Base class for baseline methods."""

    def __init__(self, backend: ModelBackend):
        self.backend = backend

    def run(self, image: Any, question: str) -> BaselineResult:
        raise NotImplementedError


class SingleVLMBaseline(BaselineRunner):
    """Single VLM baseline - direct question answering without multi-agent."""
    method = "single_vlm"

    def run(self, image: Any, question: str) -> BaselineResult:
        out = self.backend.generate(image, f"Answer the question: {question}")
        return BaselineResult(
            method=self.method,
            answer=out.text,
            agent_calls=1,
            text_tokens=len(out.text.split()),
        )


class TextOnlyMASBaseline(BaselineRunner):
    """Text-only multi-agent baseline - agents communicate only via text."""
    method = "text_only_mas"

    def run(self, image: Any, question: str) -> BaselineResult:
        agents = {
            "global": GlobalAgent(self.backend),
            "local": LocalAgent(self.backend),
            "hierarchy": HierarchyAgent(self.backend),
            "verifier": VerifierAgent(self.backend),
            "residual": ResidualAgent(self.backend),
        }
        orch = MultiAgentOrchestrator(agents, RuleController(max_steps=4))
        result = orch.run(image, question)

        stats = result.bus.stats()
        answer = result.outputs[-1][1] if result.outputs else ""

        return BaselineResult(
            method=self.method,
            answer=answer,
            communication_cost=stats["messages"] + 0.001 * stats["text_chars"],
            text_tokens=stats["text_chars"],
            struct_nodes=stats["struct_nodes"],
            agent_calls=len(result.outputs),
        )


class StaticTreeMASBaseline(BaselineRunner):
    """Static tree + MAS baseline - tree structure without RL-learned communication."""
    method = "static_tree_mas"

    def run(self, image: Any, question: str) -> BaselineResult:
        # Same as text-only but with tree context injected into prompts
        agents = {
            "global": GlobalAgent(self.backend),
            "local": LocalAgent(self.backend),
            "hierarchy": HierarchyAgent(self.backend),
            "verifier": VerifierAgent(self.backend),
            "residual": ResidualAgent(self.backend),
        }
        orch = MultiAgentOrchestrator(agents, RuleController(max_steps=5))
        result = orch.run(image, question)

        stats = result.bus.stats()
        answer = result.outputs[-1][1] if result.outputs else ""

        return BaselineResult(
            method=self.method,
            answer=answer,
            communication_cost=stats["messages"] + 0.001 * stats["text_chars"] + 0.1 * stats["struct_nodes"],
            text_tokens=stats["text_chars"],
            struct_nodes=stats["struct_nodes"],
            agent_calls=len(result.outputs),
        )
