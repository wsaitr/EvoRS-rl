"""Tests for the enhanced reward functions in rs_hmcomm.rl.rewards."""
from __future__ import annotations

from rs_hmcomm.core import (
    SceneTree,
    SceneNode,
    NodeLevel,
    NodeStatus,
    MessageBus,
    AgentMessage,
    MessageModality,
)
from rs_hmcomm.rl.rewards import (
    RewardWeights,
    team_reward,
    communication_cost,
    redundancy_cost,
    communication_cost_modality,
    structure_consistency_reward,
    evidence_reward,
    message_novelty,
    communication_reward,
)


def test_reward_weights_defaults():
    w = RewardWeights()
    assert w.structure > 0
    assert w.evidence > 0
    assert w.novelty > 0


def test_communication_cost_modality():
    bus = MessageBus()
    bus.send(AgentMessage(
        sender="global", receiver="local", spatial_level=NodeLevel.REGION,
        modalities={MessageModality.TEXT, MessageModality.STRUCT},
        text="test message", node_ids=["n1"],
    ))
    cost = communication_cost_modality(bus)
    assert cost > 0


def test_structure_consistency_reward():
    tree = SceneTree("test")
    region = SceneNode.make(NodeLevel.REGION, "apron", bbox=(0.1, 0.1, 0.8, 0.8), confidence=0.7)
    tree.add_node(region)
    group = SceneNode.make(NodeLevel.GROUP, "aircraft", parent_id=region.id, confidence=0.8)
    tree.add_node(group)
    r = structure_consistency_reward(tree)
    assert r >= 0.0


def test_evidence_reward():
    tree = SceneTree("test")
    node = SceneNode.make(NodeLevel.REGION, "apron", confidence=0.9)
    node.evidence_refs = ["crop_001"]
    tree.add_node(node)
    r = evidence_reward(tree)
    assert r > 0.0


def test_message_novelty():
    bus = MessageBus()
    msg1 = AgentMessage(sender="g", receiver="l", spatial_level=NodeLevel.REGION, text="hello")
    msg2 = AgentMessage(sender="g", receiver="l", spatial_level=NodeLevel.REGION, text="world")
    bus.messages = [msg1]
    n = message_novelty(msg2, bus)
    assert 0.0 <= n <= 1.0


def test_message_novelty_empty_history():
    bus = MessageBus()
    msg = AgentMessage(sender="g", receiver="l", spatial_level=NodeLevel.REGION, text="first")
    n = message_novelty(msg, bus)
    assert n == 1.0


def test_communication_reward():
    bus = MessageBus()
    tree = SceneTree("test")
    node = SceneNode.make(NodeLevel.REGION, "apron", confidence=0.8)
    node.evidence_refs = ["ref1"]
    tree.add_node(node)
    msg = AgentMessage(
        sender="global", receiver="local", spatial_level=NodeLevel.REGION,
        text="new message",
    )
    r = communication_reward(msg, bus, tree)
    assert isinstance(r, float)


def test_team_reward_full():
    tree = SceneTree("test")
    bus = MessageBus()
    bus.send(AgentMessage(
        sender="global", receiver="local", spatial_level=NodeLevel.REGION,
        text="test",
    ))
    r = team_reward(1.0, bus, tree=tree)
    assert isinstance(r, float)


def test_team_reward_backward_compatible():
    """Original two-argument call should still work (tree=None)."""
    bus = MessageBus()
    bus.send(AgentMessage(
        sender="global", receiver="local", spatial_level=NodeLevel.REGION,
        text="test",
    ))
    r = team_reward(1.0, bus)
    assert isinstance(r, float)


def test_original_functions_preserved():
    """communication_cost and redundancy_cost should still work as before."""
    bus = MessageBus()
    bus.send(AgentMessage(
        sender="g", receiver="l", spatial_level=NodeLevel.REGION, text="hi",
    ))
    assert communication_cost(bus) > 0
    assert redundancy_cost(bus) == 0.0  # no duplicates


def test_redundancy_cost_with_duplicates():
    bus = MessageBus()
    msg = AgentMessage(sender="g", receiver="l", spatial_level=NodeLevel.REGION, text="hi")
    bus.send(msg)
    bus.send(AgentMessage(sender="g", receiver="l", spatial_level=NodeLevel.REGION, text="hi"))
    assert redundancy_cost(bus) >= 1.0
