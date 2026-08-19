from rs_hmcomm.agents import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
from rs_hmcomm.backends.mock import MockBackend
from rs_hmcomm.controllers import RuleController
from rs_hmcomm.orchestrator import MultiAgentOrchestrator
from rs_hmcomm.core import NodeLevel

def test_stage1_dry_episode():
    b = MockBackend()
    agents = {
        "global": GlobalAgent(b),
        "local": LocalAgent(b),
        "hierarchy": HierarchyAgent(b),
        "verifier": VerifierAgent(b),
        "residual": ResidualAgent(b),
    }
    result = MultiAgentOrchestrator(agents, RuleController(max_steps=4)).run(
        {"mock": True}, "How many aircraft are visible?"
    )
    assert result.tree.query_by_level(NodeLevel.REGION)
    assert result.tree.query_by_level(NodeLevel.GROUP)
    assert result.bus.stats()["messages"] >= 2
