"""Tests for the JSONL trajectory logger."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rs_hmcomm.logging import TrajectoryLogger, EpisodeRecord, StepRecord
from rs_hmcomm.core import SceneTree, MessageBus


def test_logger_creates_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "traj.jsonl"
        logger = TrajectoryLogger(path)
        record = EpisodeRecord(episode_id="test001", question="test?", answer="yes")
        logger.log_episode(record)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["episode_id"] == "test001"


def test_logger_append():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "traj.jsonl"
        logger = TrajectoryLogger(path)
        logger.log_episode(EpisodeRecord(episode_id="ep1"))
        logger.log_episode(EpisodeRecord(episode_id="ep2"))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2


def test_logger_count():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "traj.jsonl"
        logger = TrajectoryLogger(path)
        assert logger.count == 0
        logger.log_episode(EpisodeRecord(episode_id="ep1"))
        assert logger.count == 1
        logger.log_episode(EpisodeRecord(episode_id="ep2"))
        assert logger.count == 2


def test_logger_no_append_overwrites():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "traj.jsonl"
        # First write
        logger1 = TrajectoryLogger(path)
        logger1.log_episode(EpisodeRecord(episode_id="ep1"))
        # Second logger with append=False should overwrite
        logger2 = TrajectoryLogger(path, append=False)
        logger2.log_episode(EpisodeRecord(episode_id="ep2"))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["episode_id"] == "ep2"


def test_step_record():
    step = StepRecord(step=0, agent="global", message_text="hello")
    assert step.step == 0
    assert step.agent == "global"
    assert step.message_text == "hello"


def test_episode_record_defaults():
    record = EpisodeRecord(episode_id="test")
    assert record.task_score == 0.0
    assert record.total_reward == 0.0
    assert record.steps == []
    assert record.message_stats == {}


def test_create_from_orchestrator():
    from rs_hmcomm.agents import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
    from rs_hmcomm.backends.mock import MockBackend
    from rs_hmcomm.controllers import RuleController
    from rs_hmcomm.orchestrator import MultiAgentOrchestrator

    b = MockBackend()
    agents = {
        "global": GlobalAgent(b),
        "local": LocalAgent(b),
        "hierarchy": HierarchyAgent(b),
        "verifier": VerifierAgent(b),
        "residual": ResidualAgent(b),
    }
    result = MultiAgentOrchestrator(agents, RuleController(max_steps=4)).run(
        {"mock": True}, "How many aircraft?"
    )
    record = TrajectoryLogger.create_from_orchestrator(result, question="How many aircraft?")
    assert record.question == "How many aircraft?"
    assert len(record.steps) > 0
    assert record.message_stats["messages"] > 0


def test_create_from_orchestrator_with_rewards():
    from rs_hmcomm.agents import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
    from rs_hmcomm.backends.mock import MockBackend
    from rs_hmcomm.controllers import RuleController
    from rs_hmcomm.orchestrator import MultiAgentOrchestrator

    b = MockBackend()
    agents = {
        "global": GlobalAgent(b),
        "local": LocalAgent(b),
        "hierarchy": HierarchyAgent(b),
        "verifier": VerifierAgent(b),
        "residual": ResidualAgent(b),
    }
    result = MultiAgentOrchestrator(agents, RuleController(max_steps=4)).run(
        {"mock": True}, "How many aircraft?"
    )
    rewards = {"total": 1.5, "comm_cost": 0.1, "redundancy": 0.0, "structure": 0.8, "evidence": 0.6}
    record = TrajectoryLogger.create_from_orchestrator(
        result, question="How many aircraft?", task_score=1.0, rewards=rewards,
    )
    assert record.total_reward == 1.5
    assert record.structure_reward == 0.8
    assert record.evidence_reward == 0.6
    assert record.task_score == 1.0
