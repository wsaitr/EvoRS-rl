"""Evaluation metrics for RS-HMComm experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import statistics


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Task performance
    accuracy: float = 0.0
    avg_task_score: float = 0.0

    # Communication efficiency
    avg_agent_calls: float = 0.0
    avg_communication_turns: float = 0.0
    avg_text_tokens: float = 0.0
    avg_struct_nodes: float = 0.0
    avg_latent_bytes: float = 0.0
    avg_image_crops: float = 0.0

    # Cost
    avg_communication_cost: float = 0.0
    avg_latency_ms: float = 0.0
    avg_peak_memory_mb: float = 0.0

    # Per-task breakdown
    per_task_accuracy: dict[str, float] = field(default_factory=dict)
    per_task_communication: dict[str, float] = field(default_factory=dict)

    # Score vs cost
    score_per_token: float = 0.0
    score_per_communication_unit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Accuracy: {self.accuracy:.4f}",
            f"Avg Task Score: {self.avg_task_score:.4f}",
            f"Avg Agent Calls: {self.avg_agent_calls:.2f}",
            f"Avg Text Tokens: {self.avg_text_tokens:.1f}",
            f"Avg Struct Nodes: {self.avg_struct_nodes:.1f}",
            f"Avg Comm Cost: {self.avg_communication_cost:.4f}",
            f"Score/Token: {self.score_per_token:.6f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    results: list[dict[str, Any]],
) -> EvaluationMetrics:
    """Compute aggregate metrics from a list of result dicts.

    Each result dict should have:
    - task_score: float
    - agent_calls: int
    - text_tokens: int
    - struct_nodes: int
    - latent_bytes: int
    - communication_cost: float
    - task_type: str (optional)
    - correct: bool (optional)
    """
    if not results:
        return EvaluationMetrics()

    task_scores = [r.get("task_score", 0.0) for r in results]
    agent_calls = [r.get("agent_calls", 0) for r in results]
    text_tokens = [r.get("text_tokens", 0) for r in results]
    struct_nodes = [r.get("struct_nodes", 0) for r in results]
    latent_bytes = [r.get("latent_bytes", 0) for r in results]
    comm_costs = [r.get("communication_cost", 0.0) for r in results]

    correct = [r.get("correct", False) for r in results]
    accuracy = sum(correct) / len(correct) if correct else 0.0

    avg_score = statistics.mean(task_scores) if task_scores else 0.0
    total_tokens = sum(text_tokens)
    score_per_token = avg_score / max(1, statistics.mean(text_tokens))

    avg_comm_cost = statistics.mean(comm_costs) if comm_costs else 0.0
    score_per_comm = avg_score / max(1e-8, avg_comm_cost)

    # Per-task breakdown
    per_task_scores: dict[str, list[float]] = {}
    per_task_comms: dict[str, list[float]] = {}
    for r in results:
        task_type = r.get("task_type", "unknown")
        per_task_scores.setdefault(task_type, []).append(r.get("task_score", 0.0))
        per_task_comms.setdefault(task_type, []).append(r.get("communication_cost", 0.0))

    per_task_accuracy = {k: statistics.mean(v) for k, v in per_task_scores.items()}
    per_task_communication = {k: statistics.mean(v) for k, v in per_task_comms.items()}

    return EvaluationMetrics(
        accuracy=accuracy,
        avg_task_score=avg_score,
        avg_agent_calls=statistics.mean(agent_calls) if agent_calls else 0.0,
        avg_text_tokens=statistics.mean(text_tokens) if text_tokens else 0.0,
        avg_struct_nodes=statistics.mean(struct_nodes) if struct_nodes else 0.0,
        avg_latent_bytes=statistics.mean(latent_bytes) if latent_bytes else 0.0,
        avg_communication_cost=avg_comm_cost,
        score_per_token=score_per_token,
        score_per_communication_unit=score_per_comm,
        per_task_accuracy=per_task_accuracy,
        per_task_communication=per_task_communication,
    )
