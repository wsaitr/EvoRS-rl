from .types import TaskAction, CommunicationAction
from .rewards import (
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
from .comlrl_adapter import CoMLRLTrajectoryAdapter
from .hmagrpo import (
    TrajectoryStep,
    Trajectory,
    GroupTrajectories,
    CommunicationPolicyHead,
    RolloutGenerator,
    GroupRelativeAdvantage,
    HMMAGRPOConfig,
    HMMAGRPOTrainer,
)

__all__ = [
    "TaskAction",
    "CommunicationAction",
    "RewardWeights",
    "team_reward",
    "communication_cost",
    "redundancy_cost",
    "communication_cost_modality",
    "structure_consistency_reward",
    "evidence_reward",
    "message_novelty",
    "communication_reward",
    "CoMLRLTrajectoryAdapter",
    # HM-MAGRPO
    "TrajectoryStep",
    "Trajectory",
    "GroupTrajectories",
    "CommunicationPolicyHead",
    "RolloutGenerator",
    "GroupRelativeAdvantage",
    "HMMAGRPOConfig",
    "HMMAGRPOTrainer",
]
