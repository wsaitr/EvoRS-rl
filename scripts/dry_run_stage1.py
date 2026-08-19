#!/usr/bin/env python3
from rs_hmcomm.agents import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
from rs_hmcomm.backends.mock import MockBackend
from rs_hmcomm.controllers import RuleController
from rs_hmcomm.orchestrator import MultiAgentOrchestrator

def main():
    backend = MockBackend()
    agents = {
        "global": GlobalAgent(backend),
        "local": LocalAgent(backend),
        "hierarchy": HierarchyAgent(backend),
        "verifier": VerifierAgent(backend),
        "residual": ResidualAgent(backend),
    }
    orch = MultiAgentOrchestrator(agents, RuleController(max_steps=4))
    result = orch.run({"mock": True}, "How many aircraft are in the airport region?")
    print("=== AGENT OUTPUTS ===")
    for agent, text in result.outputs:
        print(f"[{agent}] {text}")
    print("\n=== SHARED TREE ===")
    print(result.tree.compact_summary())
    print("\n=== MESSAGE STATS ===")
    print(result.bus.stats())

if __name__ == "__main__":
    main()
