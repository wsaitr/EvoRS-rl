"""HM-MAGRPO: Hierarchical Multi-Agent Group Relative Policy Optimization.

Implements the full HM-MAGRPO training algorithm for the RS-HMComm framework,
based on the paper equations:

    Eq. 1  - Communication action: a^comm_t = (r_t, l_t, m_t, p_t)
    Eq. 5  - Team reward: R = R_task + l_s*R_struct + l_e*R_evid - l_c*C_comm - l_r*C_red
    Eq. 6  - Communication cost: C_comm = alpha*N_text + beta*N_node + gamma*N_latent
    Eq. 7  - Message novelty: r^novel(m_t) = 1 - max_{m in H_t} sim(m_t, m)
    Eq. 8  - Communication reward: r^comm_t = eta1*DeltaE_t + eta2*r^novel(m_t) - eta3*cost(m_t)
    Eq. 9  - Group-relative advantage: hat{A}^comm_t = hat{A}^team + rho*norm(r^comm_t)
    Eq. 10 - HM-MAGRPO loss (dual clipping):
             L = E[min(r_task * A_team, clip(r_task, 1+-eps_t) * A_team)
                 + kappa * min(r_comm * A_comm, clip(r_comm, 1+-eps_c) * A_comm)]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum
import math
import uuid

from rs_hmcomm.core import SceneTree, MessageBus, NodeLevel, MessageModality
from rs_hmcomm.rl.types import TaskAction, CommunicationAction


# ---------------------------------------------------------------------------
# Trajectory data classes
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """One step in a rollout trajectory.

    Captures the agent identity, both task and communication actions taken,
    the log-probabilities under the old policy, and the reward signals
    received at this step.
    """
    step: int
    agent_id: str
    task_action: TaskAction
    comm_action: CommunicationAction
    logprob_task: float = 1.0
    logprob_comm: float = 1.0
    reward: float = 0.0
    team_reward: float = 0.0
    comm_reward: float = 0.0
    tree_snapshot: dict[str, Any] = field(default_factory=dict)
    message_text: str = ""


@dataclass
class Trajectory:
    """Full rollout trajectory for one episode."""
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    answer: str = ""
    task_score: float = 0.0
    steps: list[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    communication_cost: float = 0.0


@dataclass
class GroupTrajectories:
    """Group of trajectories for the same question (for group-relative advantage).

    In MAGRPO, multiple rollouts are collected for the same input so that
    advantages can be estimated relative to the group mean/std.
    """
    question: str
    ground_truth: str
    trajectories: list[Trajectory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Communication policy head
# ---------------------------------------------------------------------------

class CommunicationPolicyHead:
    """Policy head for communication actions.

    In a full neural implementation this would be a learned network that
    maps (agent_state, tree_state, bus_state) -> distribution over
    communication actions.  For now it provides a rule-based / uniform
    sampling interface so that the training loop can be exercised
    end-to-end.

    The action space is the Cartesian product of:
      - RECIPIENTS  (target agent)
      - LEVELS      (spatial level to report on)
      - MODALITIES  (combination of text / struct / latent)
    """

    RECIPIENTS = ("global", "local", "hierarchy", "verifier", "residual")
    LEVELS = (NodeLevel.SCENE, NodeLevel.REGION, NodeLevel.GROUP, NodeLevel.OBJECT)
    MODALITIES = (
        (MessageModality.TEXT,),
        (MessageModality.STRUCT,),
        (MessageModality.LATENT,),
        (MessageModality.TEXT, MessageModality.STRUCT),
        (MessageModality.STRUCT, MessageModality.LATENT),
        (MessageModality.TEXT, MessageModality.STRUCT, MessageModality.LATENT),
    )

    def __init__(self) -> None:
        self._action_space_size = (
            len(self.RECIPIENTS) * len(self.LEVELS) * len(self.MODALITIES)
        )
        self._logprobs: dict[str, float] = {}

    def sample_action(
        self,
        agent_id: str,
        tree: SceneTree,
        bus: MessageBus,
        step: int,
    ) -> tuple[CommunicationAction, float]:
        """Sample a communication action.  Returns ``(action, logprob)``.

        The current implementation uses simple heuristics based on the
        scene-tree state to pick a *reasonable* action, and returns the
        uniform log-probability.  A learned policy would replace the
        body of this method.
        """
        import random

        regions = tree.query_by_level(NodeLevel.REGION)
        groups = tree.query_by_level(NodeLevel.GROUP)

        # Choose recipient based on tree state
        if not regions:
            recipient = "global"
            level = NodeLevel.SCENE
            modality = (MessageModality.TEXT,)
        elif not groups:
            recipient = "local"
            level = NodeLevel.REGION
            modality = (MessageModality.TEXT, MessageModality.STRUCT)
        else:
            recipient = random.choice(["hierarchy", "verifier"])
            level = NodeLevel.GROUP if groups else NodeLevel.REGION
            modality = (MessageModality.TEXT, MessageModality.STRUCT)

        # Determine payload from the target-level nodes
        target_nodes = tree.query_by_level(level)
        payload_ids = tuple(n.id for n in target_nodes[:3])

        # Pick a task action aligned with the recipient role
        task_action = TaskAction.INSPECT
        if recipient == "verifier":
            task_action = TaskAction.VERIFY
        elif recipient == "hierarchy":
            task_action = TaskAction.AGGREGATE

        action = CommunicationAction(
            task_action=task_action,
            recipient=recipient,
            spatial_level=level,
            modality=modality,
            payload_node_ids=payload_ids,
        )

        # Uniform log-probability over the (currently explored) action space
        logprob = -math.log(self._action_space_size)
        return action, logprob

    def log_probability(self, action: CommunicationAction) -> float:
        """Compute (uniform) log probability of an action."""
        return -math.log(self._action_space_size)


# ---------------------------------------------------------------------------
# Rollout generator
# ---------------------------------------------------------------------------

class RolloutGenerator:
    """Generates rollout trajectories by running multi-agent episodes.

    The generator wraps an orchestrator *factory* (a callable returning a
    fresh ``MultiAgentOrchestrator``) so that every trajectory starts from
    a clean state.
    """

    def __init__(
        self,
        orchestrator_factory: Callable,
        comm_policy: CommunicationPolicyHead | None = None,
        max_steps: int = 5,
    ) -> None:
        self.orchestrator_factory = orchestrator_factory
        self.comm_policy = comm_policy or CommunicationPolicyHead()
        self.max_steps = max_steps

    def generate_trajectory(
        self,
        image: Any,
        question: str,
        answer: str = "",
    ) -> Trajectory:
        """Run one episode and collect a full trajectory."""
        from rs_hmcomm.orchestrator import MultiAgentOrchestrator, EpisodeResult
        from rs_hmcomm.rl.rewards import team_reward, communication_cost

        orch = self.orchestrator_factory()
        result = orch.run(image, question)

        traj = Trajectory(
            question=question,
            answer=answer,
        )

        # Build trajectory steps from the orchestrator outputs
        for i, (agent_name, text) in enumerate(result.outputs):
            # Ask the policy head for a comm action (and its log-prob)
            action, logprob = self.comm_policy.sample_action(
                agent_id=agent_name,
                tree=result.tree,
                bus=result.bus,
                step=i,
            )
            step_record = TrajectoryStep(
                step=i,
                agent_id=agent_name,
                task_action=action.task_action,
                comm_action=action,
                logprob_task=1.0,
                logprob_comm=logprob,
                message_text=text,
                tree_snapshot=result.tree.to_dict(),
            )
            traj.steps.append(step_record)

        # Compute episode-level rewards
        traj.communication_cost = communication_cost(result.bus)
        task_score = 1.0 if result.stopped_by else 0.0
        traj.total_reward = team_reward(task_score, result.bus, result.tree)
        traj.task_score = task_score

        return traj

    def generate_group(
        self,
        image: Any,
        question: str,
        ground_truth: str = "",
        group_size: int = 4,
    ) -> GroupTrajectories:
        """Generate multiple trajectories for the same question.

        The group of rollouts is used to compute group-relative advantages
        (MAGRPO core idea).
        """
        group = GroupTrajectories(question=question, ground_truth=ground_truth)
        for _ in range(group_size):
            traj = self.generate_trajectory(image, question, ground_truth)
            group.trajectories.append(traj)
        return group


# ---------------------------------------------------------------------------
# Group-relative advantage estimator
# ---------------------------------------------------------------------------

class GroupRelativeAdvantage:
    """Computes group-relative advantages for both task and communication.

    Implements Eq. (9)::

        hat{A}^comm_t = hat{A}^team + rho * norm(r^comm_t)

    where ``hat{A}^team`` is the normalised group return and
    ``norm(r^comm_t)`` is the mean normalised communication reward
    within the trajectory.
    """

    def __init__(self, rho: float = 0.5) -> None:
        self.rho = rho

    def compute(self, group: GroupTrajectories) -> list[dict[str, float]]:
        """Compute advantages for each trajectory in the group.

        Returns a list of dicts, one per trajectory, each containing:
          - ``team_advantage``:  normalised group return
          - ``comm_advantage``:  team_advantage + rho * mean_comm_reward
        """
        if not group.trajectories:
            return []

        rewards = [t.total_reward for t in group.trajectories]
        mean_r = sum(rewards) / len(rewards)
        std_r = max(
            1e-8,
            math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards)),
        )

        # Normalise comm rewards across all trajectories for stable scaling
        all_comm = [s.comm_reward for t in group.trajectories for s in t.steps]
        mean_comm = sum(all_comm) / max(1, len(all_comm))
        std_comm = max(
            1e-8,
            math.sqrt(
                sum((c - mean_comm) ** 2 for c in all_comm) / max(1, len(all_comm))
            ),
        )

        advantages: list[dict[str, float]] = []
        for traj in group.trajectories:
            # Normalised team return
            norm_r = (traj.total_reward - mean_r) / std_r

            # Mean comm reward within this trajectory, normalised
            traj_comm = [s.comm_reward for s in traj.steps]
            traj_mean_comm = sum(traj_comm) / max(1, len(traj_comm))
            norm_comm = (traj_mean_comm - mean_comm) / std_comm

            advantages.append({
                "team_advantage": norm_r,
                "comm_advantage": norm_r + self.rho * norm_comm,
            })

        return advantages


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HMMAGRPOConfig:
    """Configuration for HM-MAGRPO training.

    Groups parameters by their role:
      - Clipping parameters (epsilon_task, epsilon_comm, kappa)
      - Reward weights (lambda_*, eta_*)
      - Communication cost weights (alpha, beta, gamma)
      - Group-relative parameters (rho, group_size)
      - Training hyper-parameters (learning_rate, max_epochs, batch_size)
    """
    # Clipping params
    epsilon_task: float = 0.2
    epsilon_comm: float = 0.2
    kappa: float = 0.5  # weight for comm loss vs task loss

    # Reward weights (Eq. 5)
    lambda_struct: float = 0.1
    lambda_evid: float = 0.1
    lambda_cost: float = 0.02
    lambda_redundancy: float = 0.05

    # Communication cost weights (Eq. 6)
    alpha_text: float = 1.0
    beta_node: float = 0.1
    gamma_latent: float = 0.25

    # Communication reward weights (Eq. 8)
    eta_evidence: float = 0.3
    eta_novelty: float = 0.3
    eta_cost: float = 0.2

    # Group relative (Eq. 9)
    rho: float = 0.5
    group_size: int = 4

    # Training
    learning_rate: float = 1e-5
    max_epochs: int = 3
    batch_size: int = 8


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HMMAGRPOTrainer:
    """Full HM-MAGRPO training loop (Algorithm 1 in the paper).

    The trainer orchestrates:
      1. Rollout generation (via :class:`RolloutGenerator`)
      2. Group-relative advantage estimation (via :class:`GroupRelativeAdvantage`)
      3. Dual-clip loss computation (Eq. 10)
      4. Logging of training metrics

    In a full implementation the ``compute_loss`` gradients would be fed
    to an optimiser that updates both the task VLM and the communication
    policy head.  The current implementation computes the loss values
    correctly but does not perform the gradient step (no neural net is
    wired up yet).
    """

    def __init__(self, config: HMMAGRPOConfig | None = None) -> None:
        self.config = config or HMMAGRPOConfig()
        self.comm_policy = CommunicationPolicyHead()
        self.rollout_gen: RolloutGenerator | None = None
        self.gra = GroupRelativeAdvantage(rho=self.config.rho)
        self._history: list[dict[str, Any]] = []

    # -- setup --------------------------------------------------------------

    def setup_rollout_generator(self, orchestrator_factory: Callable) -> None:
        """Set up the rollout generator with an orchestrator factory."""
        self.rollout_gen = RolloutGenerator(
            orchestrator_factory=orchestrator_factory,
            comm_policy=self.comm_policy,
        )

    # -- loss (Eq. 10) ------------------------------------------------------

    def compute_loss(
        self,
        group: GroupTrajectories,
    ) -> dict[str, float]:
        """Compute the HM-MAGRPO loss for a group of trajectories.

        Implements the dual-clip objective (Eq. 10)::

            L = E[ min(r_task * A_team, clip(r_task, 1+-eps_t) * A_team)
                 + kappa * min(r_comm * A_comm, clip(r_comm, 1+-eps_c) * A_comm) ]

        The ratio ``r`` is the importance-sampling ratio (pi_new / pi_old).
        In this initial implementation we store the log-prob from the
        rollout and use it as a proxy for the ratio; a full implementation
        would re-evaluate log-probs under the current policy.
        """
        advantages = self.gra.compute(group)

        total_loss = 0.0
        task_loss_sum = 0.0
        comm_loss_sum = 0.0

        for traj, adv in zip(group.trajectories, advantages):
            for step in traj.steps:
                # --- task branch ---
                r_task = step.logprob_task  # IS ratio (proxy)
                a_task = adv["team_advantage"]

                unclipped_task = r_task * a_task
                clipped_task = (
                    max(min(r_task, 1.0 + self.config.epsilon_task),
                        1.0 - self.config.epsilon_task)
                    * a_task
                )
                task_loss = min(unclipped_task, clipped_task)

                # --- communication branch ---
                r_comm = step.logprob_comm
                a_comm = adv["comm_advantage"]

                unclipped_comm = r_comm * a_comm
                clipped_comm = (
                    max(min(r_comm, 1.0 + self.config.epsilon_comm),
                        1.0 - self.config.epsilon_comm)
                    * a_comm
                )
                comm_loss = min(unclipped_comm, clipped_comm)

                # --- combined (negated for gradient descent) ---
                step_loss = -(task_loss + self.config.kappa * comm_loss)
                total_loss += step_loss
                task_loss_sum += task_loss
                comm_loss_sum += comm_loss

        n_steps = max(1, sum(len(t.steps) for t in group.trajectories))
        return {
            "total_loss": total_loss / n_steps,
            "task_loss": task_loss_sum / n_steps,
            "comm_loss": comm_loss_sum / n_steps,
            "n_trajectories": len(group.trajectories),
            "n_steps": n_steps,
        }

    # -- training step / epoch ----------------------------------------------

    def train_step(
        self,
        image: Any,
        question: str,
        ground_truth: str = "",
    ) -> dict[str, float]:
        """One training step: generate group, compute loss.

        Corresponds to one iteration of the outer loop in Algorithm 1.
        """
        if self.rollout_gen is None:
            raise RuntimeError("Call setup_rollout_generator first")

        # Generate group of trajectories
        group = self.rollout_gen.generate_group(
            image=image,
            question=question,
            ground_truth=ground_truth,
            group_size=self.config.group_size,
        )

        # Compute loss
        loss_info = self.compute_loss(group)
        loss_info["question"] = question

        self._history.append(loss_info)
        return loss_info

    def train_epoch(
        self,
        dataset: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Train for one epoch over the dataset."""
        total_loss = 0.0
        n = 0
        for item in dataset:
            info = self.train_step(
                image=item.get("image"),
                question=item.get("question", ""),
                ground_truth=item.get("answer", ""),
            )
            total_loss += info["total_loss"]
            n += 1

        return {
            "epoch_loss": total_loss / max(1, n),
            "n_samples": n,
        }

    # -- introspection ------------------------------------------------------

    @property
    def training_history(self) -> list[dict[str, Any]]:
        """Return the full training history (one dict per training step)."""
        return self._history
