"""Reward functions for RS-HMComm multi-agent RL training.

Implements the five-term reward from the paper (Eq. 5):
    R = R_task + lambda_s * R_struct + lambda_e * R_evid - lambda_c * C_comm - lambda_r * C_red

Plus helper functions for modality-weighted cost, structure consistency,
evidence reward, and message novelty.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from rs_hmcomm.core import (
    MessageBus,
    AgentMessage,
    MessageModality,
    NodeLevel,
    SceneTree,
    SceneNode,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Level ordering for parent-child consistency checks
# ---------------------------------------------------------------------------
_LEVEL_ORDER: dict[NodeLevel, int] = {
    NodeLevel.SCENE: 0,
    NodeLevel.REGION: 1,
    NodeLevel.GROUP: 2,
    NodeLevel.OBJECT: 3,
}

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Weights for the five-term team reward (paper Eq. 5).

    R = w.task * R_task
      + w.structure * R_struct
      + w.evidence * R_evid
      - w.communication_cost * C_comm
      - w.redundancy * C_red
    """
    task: float = 1.0
    communication_cost: float = 0.02
    redundancy: float = 0.05
    structure: float = 0.1
    evidence: float = 0.1
    novelty: float = 0.05


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def communication_cost(bus: MessageBus) -> float:
    """Scalar communication cost from bus statistics.

    C_comm = N_msg + 0.001 * N_text_chars + 0.10 * N_struct_nodes + 0.25 * N_latent
    """
    s = bus.stats()
    return s["messages"] + 0.001 * s["text_chars"] + 0.10 * s["struct_nodes"] + 0.25 * s["latent_handles"]


def redundancy_cost(bus: MessageBus) -> float:
    """Count duplicate messages (same receiver, same nodes, same text)."""
    payloads = [
        (m.receiver, tuple(sorted(m.node_ids)), m.text.strip().lower())
        for m in bus.messages
    ]
    counts = Counter(payloads)
    return float(sum(max(0, n - 1) for n in counts.values()))


def communication_cost_modality(
    bus: MessageBus,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.25,
) -> float:
    """Cost weighted by modality: alpha * N_text + beta * N_node + gamma * N_latent.

    For each message we count:
      - TEXT  -> alpha  (if TEXT in modalities)
      - STRUCT -> beta * len(node_ids)
      - LATENT -> gamma * len(latent_handles)
    """
    total = 0.0
    for m in bus.messages:
        if MessageModality.TEXT in m.modalities:
            total += alpha
        if MessageModality.STRUCT in m.modalities:
            total += beta * len(m.node_ids)
        if MessageModality.LATENT in m.modalities:
            total += gamma * len(m.latent_handles)
    return total


# ---------------------------------------------------------------------------
# Structure consistency reward
# ---------------------------------------------------------------------------

def structure_consistency_reward(tree: SceneTree) -> float:
    """Check parent-child level consistency in the scene tree.

    Reward is the fraction of edges where the child level is exactly one
    step deeper than the parent level in the canonical ordering
    (SCENE < REGION < GROUP < OBJECT).  Returns a value in [0, 1].
    """
    nodes = list(tree._nodes.values())
    if len(nodes) <= 1:
        return 1.0  # trivially consistent

    valid_edges = 0
    total_edges = 0
    for node in nodes:
        if node.parent_id is None:
            continue
        if node.parent_id not in tree._nodes:
            continue
        parent = tree._nodes[node.parent_id]
        total_edges += 1
        parent_depth = _LEVEL_ORDER.get(parent.level, -1)
        child_depth = _LEVEL_ORDER.get(node.level, -1)
        if child_depth == parent_depth + 1:
            valid_edges += 1

    if total_edges == 0:
        return 1.0
    return valid_edges / total_edges


# ---------------------------------------------------------------------------
# Evidence reward
# ---------------------------------------------------------------------------

def evidence_reward(tree: SceneTree, threshold: float = 0.5) -> float:
    """Reward nodes with high confidence that also have evidence references.

    For each non-scene node with confidence >= threshold, we give +1 if
    it has at least one evidence_ref, 0 otherwise.  Normalised by the
    number of high-confidence nodes.  Returns a value in [0, 1].
    """
    hc_nodes = [
        n for n in tree._nodes.values()
        if n.level != NodeLevel.SCENE and n.confidence >= threshold
    ]
    if not hc_nodes:
        return 0.0
    with_evidence = sum(1 for n in hc_nodes if len(n.evidence_refs) > 0)
    return with_evidence / len(hc_nodes)


# ---------------------------------------------------------------------------
# Message novelty
# ---------------------------------------------------------------------------

def message_novelty(msg: AgentMessage, history: MessageBus) -> float:
    """Compute 1 - max_similarity to prior messages in history.

    Similarity is based on text overlap (SequenceMatcher ratio).
    If history is empty, novelty is 1.0.
    Returns a value in [0, 1].
    """
    if not history.messages:
        return 1.0

    text = msg.text.strip().lower()
    if not text:
        return 0.0

    max_sim = 0.0
    for prior in history.messages:
        prior_text = prior.text.strip().lower()
        if not prior_text:
            continue
        sim = SequenceMatcher(None, text, prior_text).ratio()
        max_sim = max(max_sim, sim)

    return 1.0 - max_sim


# ---------------------------------------------------------------------------
# Communication reward (composite)
# ---------------------------------------------------------------------------

def communication_reward(
    msg: AgentMessage,
    bus: MessageBus,
    tree: SceneTree,
    weights: RewardWeights | None = None,
) -> float:
    """Combine novelty, evidence gain, and cost into a single reward signal.

    R_comm = w.novelty * novelty(msg, bus)
           + w.evidence * evidence_gain(tree)
           - w.communication_cost * cost(bus)
    """
    w = weights or RewardWeights()
    novelty = message_novelty(msg, bus)
    evid = evidence_reward(tree)
    cost = communication_cost(bus)
    return w.novelty * novelty + w.evidence * evid - w.communication_cost * cost


# ---------------------------------------------------------------------------
# Team reward (paper Eq. 5)
# ---------------------------------------------------------------------------

def team_reward(
    task_score: float,
    bus: MessageBus,
    tree: SceneTree | None = None,
    weights: RewardWeights | None = None,
) -> float:
    """Five-term team reward from paper Eq. (5).

    R = R_task + lambda_s * R_struct + lambda_e * R_evid
        - lambda_c * C_comm - lambda_r * C_red

    If *tree* is None the structure and evidence terms are 0 (backward
    compatible with the original two-argument signature).
    """
    w = weights or RewardWeights()

    r_task = w.task * task_score

    # Structure & evidence require a tree
    if tree is not None:
        r_struct = w.structure * structure_consistency_reward(tree)
        r_evid = w.evidence * evidence_reward(tree)
    else:
        r_struct = 0.0
        r_evid = 0.0

    c_comm = w.communication_cost * communication_cost(bus)
    c_red = w.redundancy * redundancy_cost(bus)

    return r_task + r_struct + r_evid - c_comm - c_red
