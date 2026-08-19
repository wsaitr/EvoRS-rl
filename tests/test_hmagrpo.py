"""Tests for the HM-MAGRPO training algorithm.

Verifies the core building blocks:
  - CommunicationPolicyHead sampling
  - GroupRelativeAdvantage computation
  - HMMAGRPOConfig defaults
  - HMMAGRPOTrainer initialisation and loss computation
  - RolloutGenerator end-to-end with mock agents
"""
from __future__ import annotations

import pytest

from rs_hmcomm.rl.hmagrpo import (
    Trajectory,
    TrajectoryStep,
    GroupTrajectories,
    CommunicationPolicyHead,
    RolloutGenerator,
    GroupRelativeAdvantage,
    HMMAGRPOConfig,
    HMMAGRPOTrainer,
)
from rs_hmcomm.core import NodeLevel, MessageModality
from rs_hmcomm.rl.types import TaskAction, CommunicationAction


# ---------------------------------------------------------------------------
# CommunicationPolicyHead
# ---------------------------------------------------------------------------

def test_communication_policy_head_sample():
    """Sampling a communication action returns a valid action and log-prob."""
    from rs_hmcomm.core import SceneTree, MessageBus

    head = CommunicationPolicyHead()
    tree = SceneTree("test")
    bus = MessageBus()
    action, logprob = head.sample_action("global", tree, bus, step=0)
    assert isinstance(action, CommunicationAction)
    assert isinstance(logprob, float)


def test_communication_policy_head_logprob_negative():
    """Log-probability should be negative (log of a value in (0, 1])."""
    head = CommunicationPolicyHead()
    action = CommunicationAction(
        task_action=TaskAction.INSPECT,
        recipient="global",
        spatial_level=NodeLevel.SCENE,
        modality=(MessageModality.TEXT,),
    )
    lp = head.log_probability(action)
    assert lp < 0.0


# ---------------------------------------------------------------------------
# GroupRelativeAdvantage
# ---------------------------------------------------------------------------

def test_group_relative_advantage():
    """GRA should return one advantage dict per trajectory."""
    gra = GroupRelativeAdvantage(rho=0.5)
    group = GroupTrajectories(question="test?", ground_truth="yes")
    for i in range(4):
        traj = Trajectory(
            question="test?",
            total_reward=float(i),
            steps=[
                TrajectoryStep(
                    step=0,
                    agent_id="global",
                    task_action=TaskAction.INSPECT,
                    comm_action=CommunicationAction(
                        TaskAction.INSPECT, "", NodeLevel.SCENE,
                        (MessageModality.TEXT,),
                    ),
                    comm_reward=float(i) * 0.1,
                ),
            ],
        )
        group.trajectories.append(traj)

    advantages = gra.compute(group)
    assert len(advantages) == 4
    assert "team_advantage" in advantages[0]
    assert "comm_advantage" in advantages[0]


def test_group_relative_advantage_empty():
    """GRA on an empty group should return an empty list."""
    gra = GroupRelativeAdvantage()
    group = GroupTrajectories(question="?", ground_truth="")
    assert gra.compute(group) == []


def test_group_relative_advantage_zero_std():
    """GRA with all-equal rewards should not crash (std clamped to 1e-8)."""
    gra = GroupRelativeAdvantage(rho=0.5)
    group = GroupTrajectories(question="test?", ground_truth="yes")
    for _ in range(3):
        traj = Trajectory(
            question="test?",
            total_reward=1.0,
            steps=[
                TrajectoryStep(
                    step=0,
                    agent_id="global",
                    task_action=TaskAction.INSPECT,
                    comm_action=CommunicationAction(
                        TaskAction.INSPECT, "", NodeLevel.SCENE,
                        (MessageModality.TEXT,),
                    ),
                    comm_reward=0.5,
                ),
            ],
        )
        group.trajectories.append(traj)

    advantages = gra.compute(group)
    assert len(advantages) == 3
    # All rewards are equal so normalised advantages should be ~0
    for adv in advantages:
        assert abs(adv["team_advantage"]) < 1e-6


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_hmgrpo_config_defaults():
    config = HMMAGRPOConfig()
    assert config.epsilon_task == 0.2
    assert config.kappa == 0.5
    assert config.group_size == 4
    assert config.rho == 0.5
    assert config.alpha_text == 1.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def test_hmgrpo_trainer_init():
    trainer = HMMAGRPOTrainer()
    assert trainer.config is not None
    assert trainer.comm_policy is not None
    assert trainer.rollout_gen is None


def test_hmgrpo_trainer_train_step_without_rollout_raises():
    trainer = HMMAGRPOTrainer()
    with pytest.raises(RuntimeError, match="setup_rollout_generator"):
        trainer.train_step(image={}, question="test?")


def test_hmgrpo_compute_loss():
    """Trainer should compute finite loss values on a simple group."""
    trainer = HMMAGRPOTrainer(HMMAGRPOConfig(group_size=2))
    group = GroupTrajectories(question="test?", ground_truth="yes")
    for i in range(2):
        traj = Trajectory(
            question="test?",
            total_reward=float(i),
            steps=[
                TrajectoryStep(
                    step=0,
                    agent_id="global",
                    task_action=TaskAction.INSPECT,
                    comm_action=CommunicationAction(
                        TaskAction.INSPECT, "", NodeLevel.SCENE,
                        (MessageModality.TEXT,),
                    ),
                    logprob_task=0.5,
                    logprob_comm=0.5,
                ),
            ],
        )
        group.trajectories.append(traj)

    loss_info = trainer.compute_loss(group)
    assert "total_loss" in loss_info
    assert "task_loss" in loss_info
    assert "comm_loss" in loss_info
    assert loss_info["n_trajectories"] == 2
    assert loss_info["n_steps"] == 2
    # Losses should be finite numbers
    import math
    assert math.isfinite(loss_info["total_loss"])
    assert math.isfinite(loss_info["task_loss"])
    assert math.isfinite(loss_info["comm_loss"])


# ---------------------------------------------------------------------------
# RolloutGenerator (end-to-end with mock agents)
# ---------------------------------------------------------------------------

def test_rollout_generator():
    """Full trajectory generation with mock backend should work end-to-end."""
    from rs_hmcomm.agents import (
        GlobalAgent, LocalAgent, HierarchyAgent,
        VerifierAgent, ResidualAgent,
    )
    from rs_hmcomm.backends.mock import MockBackend
    from rs_hmcomm.controllers import RuleController
    from rs_hmcomm.orchestrator import MultiAgentOrchestrator

    def factory():
        b = MockBackend()
        agents = {
            "global": GlobalAgent(b),
            "local": LocalAgent(b),
            "hierarchy": HierarchyAgent(b),
            "verifier": VerifierAgent(b),
            "residual": ResidualAgent(b),
        }
        return MultiAgentOrchestrator(agents, RuleController(max_steps=4))

    gen = RolloutGenerator(factory)
    traj = gen.generate_trajectory({"mock": True}, "How many aircraft?")
    assert len(traj.steps) > 0
    assert isinstance(traj.total_reward, float)
    assert isinstance(traj.communication_cost, float)


def test_rollout_generator_group():
    """Group generation should produce the requested number of trajectories."""
    from rs_hmcomm.agents import (
        GlobalAgent, LocalAgent, HierarchyAgent,
        VerifierAgent, ResidualAgent,
    )
    from rs_hmcomm.backends.mock import MockBackend
    from rs_hmcomm.controllers import RuleController
    from rs_hmcomm.orchestrator import MultiAgentOrchestrator

    def factory():
        b = MockBackend()
        agents = {
            "global": GlobalAgent(b),
            "local": LocalAgent(b),
            "hierarchy": HierarchyAgent(b),
            "verifier": VerifierAgent(b),
            "residual": ResidualAgent(b),
        }
        return MultiAgentOrchestrator(agents, RuleController(max_steps=3))

    gen = RolloutGenerator(factory)
    group = gen.generate_group(
        image={"mock": True},
        question="How many aircraft?",
        ground_truth="5",
        group_size=3,
    )
    assert len(group.trajectories) == 3
    for traj in group.trajectories:
        assert isinstance(traj, Trajectory)
        assert traj.question == "How many aircraft?"


# ---------------------------------------------------------------------------
# Full trainer end-to-end
# ---------------------------------------------------------------------------

def test_trainer_full_pipeline():
    """Trainer.train_step should run the full HM-MAGRPO pipeline."""
    from rs_hmcomm.agents import (
        GlobalAgent, LocalAgent, HierarchyAgent,
        VerifierAgent, ResidualAgent,
    )
    from rs_hmcomm.backends.mock import MockBackend
    from rs_hmcomm.controllers import RuleController
    from rs_hmcomm.orchestrator import MultiAgentOrchestrator

    def factory():
        b = MockBackend()
        agents = {
            "global": GlobalAgent(b),
            "local": LocalAgent(b),
            "hierarchy": HierarchyAgent(b),
            "verifier": VerifierAgent(b),
            "residual": ResidualAgent(b),
        }
        return MultiAgentOrchestrator(agents, RuleController(max_steps=3))

    config = HMMAGRPOConfig(group_size=2)
    trainer = HMMAGRPOTrainer(config)
    trainer.setup_rollout_generator(factory)

    info = trainer.train_step(
        image={"mock": True},
        question="How many aircraft?",
        ground_truth="5",
    )
    assert "total_loss" in info
    assert "question" in info
    assert len(trainer.training_history) == 1
