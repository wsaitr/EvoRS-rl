"""JSONL Trajectory Logger for RS-HMComm.

Records per-episode trajectory data for analysis and RL training.
Each episode is written as a single JSON line (JSONL format), making it
easy to stream, append, and parse incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from pathlib import Path
import json
import uuid
from datetime import datetime


@dataclass
class StepRecord:
    """One step in an episode."""
    step: int
    agent: str
    task_action: str = ""
    comm_action: dict[str, Any] = field(default_factory=dict)  # {recipient, level, modality, payload_node_ids}
    tree_snapshot: dict[str, Any] = field(default_factory=dict)
    message_text: str = ""
    message_node_ids: list[str] = field(default_factory=list)
    reward: float = 0.0


@dataclass
class EpisodeRecord:
    """Full episode record."""
    episode_id: str
    image_id: str = ""
    question: str = ""
    answer: str = ""
    task_score: float = 0.0
    total_reward: float = 0.0
    communication_cost: float = 0.0
    redundancy_cost: float = 0.0
    structure_reward: float = 0.0
    evidence_reward: float = 0.0
    steps: list[StepRecord] = field(default_factory=list)
    message_stats: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    """Writes episode records to a JSONL file."""

    def __init__(self, output_path: str | Path, append: bool = True) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = "a" if append else "w"
        self._count = 0

    def log_episode(self, record: EpisodeRecord) -> None:
        """Write one episode record as a JSON line."""
        with open(self.output_path, self.mode, encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self.mode = "a"  # switch to append after first write
        self._count += 1

    @staticmethod
    def create_from_orchestrator(
        episode_result,  # EpisodeResult from orchestrator
        question: str = "",
        answer: str = "",
        task_score: float = 0.0,
        rewards: dict[str, float] | None = None,
    ) -> EpisodeRecord:
        """Create an EpisodeRecord from an orchestrator EpisodeResult."""
        record = EpisodeRecord(
            episode_id=uuid.uuid4().hex[:16],
            question=question,
            answer=answer,
            task_score=task_score,
        )
        if rewards:
            record.total_reward = rewards.get("total", 0.0)
            record.communication_cost = rewards.get("comm_cost", 0.0)
            record.redundancy_cost = rewards.get("redundancy", 0.0)
            record.structure_reward = rewards.get("structure", 0.0)
            record.evidence_reward = rewards.get("evidence", 0.0)

        # Convert tree and messages
        record.message_stats = episode_result.bus.stats()
        for i, (agent, text) in enumerate(episode_result.outputs):
            step = StepRecord(step=i, agent=agent, message_text=text)
            record.steps.append(step)

        return record

    @property
    def count(self) -> int:
        return self._count
