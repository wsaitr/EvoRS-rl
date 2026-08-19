"""Training configuration for Stage 2 HM-MAGRPO."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import yaml


@dataclass
class TrainerConfig:
    """Full trainer configuration."""
    # Framework
    framework: str = "comlrl"
    algorithm: str = "magrpo"

    # Model
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    training_mode: str = "lora"
    lora_rank: int = 16
    lora_alpha: int = 32

    # Data
    dataset: str = "vrsbench"
    train_samples: int = 2000
    eval_samples: int = 500

    # Training
    max_agent_steps: int = 4
    group_size: int = 4
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_epochs: int = 3
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])

    # Communication action
    comm_recipient: bool = True
    comm_spatial_level: bool = True
    comm_modality: bool = True
    comm_payload_nodes: bool = True

    # Reward
    reward_task: float = 1.0
    reward_communication_cost: float = 0.02
    reward_redundancy: float = 0.05
    reward_structure: float = 0.1
    reward_evidence: float = 0.1

    # Channels
    structured_channel: bool = True
    latent_channel: bool = False

    # Output
    output_dir: str = "outputs/stage2"
    log_interval: int = 10
    checkpoint_interval: int = 100

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainerConfig:
        """Load config from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        from dataclasses import asdict
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
