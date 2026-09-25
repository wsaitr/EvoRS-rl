"""Small, deterministic task-metric contracts for Phase 0."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, Mapping

from ._validation import strict_dict


def normalize_short_answer(value: str) -> str:
    """Normalize a short VQA answer without task-specific guessing."""

    if not isinstance(value, str):
        raise ValueError("answer must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w.\-]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    """Evaluation-only labels, deliberately kept outside AgentObservation."""

    sample_id: str
    references: tuple[str, ...]
    task_type: str = "vqa"

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.references or any(
            not isinstance(item, str) or not item.strip() for item in self.references
        ):
            raise ValueError("references must contain non-empty strings")
        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ValueError("task_type must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "references": list(self.references),
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationTarget":
        values = strict_dict(
            data,
            cls.__name__,
            required={"sample_id", "references", "task_type"},
        )
        references = values["references"]
        if not isinstance(references, list):
            raise ValueError("references must be a list")
        return cls(
            sample_id=values["sample_id"],
            references=tuple(references),
            task_type=values["task_type"],
        )


@dataclass(frozen=True, slots=True)
class TaskScore:
    sample_id: str
    metric: str
    value: float
    correct: bool
    task_type: str = "vqa"

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.sample_id, self.metric, self.task_type)
        ):
            raise ValueError("sample_id, metric, and task_type must be non-empty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("value must be a number between zero and one")
        if not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("value must be between zero and one")
        if not isinstance(self.correct, bool):
            raise ValueError("correct must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metric": self.metric,
            "value": float(self.value),
            "correct": self.correct,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskScore":
        values = strict_dict(
            data,
            cls.__name__,
            required={"sample_id", "metric", "value", "correct", "task_type"},
        )
        if not isinstance(values["correct"], bool):
            raise ValueError("correct must be a boolean")
        return cls(
            sample_id=values["sample_id"],
            metric=values["metric"],
            value=values["value"],
            correct=values["correct"],
            task_type=values["task_type"],
        )


def exact_match_score(prediction: str, target: EvaluationTarget) -> TaskScore:
    normalized_prediction = normalize_short_answer(prediction)
    normalized_references = {normalize_short_answer(item) for item in target.references}
    correct = normalized_prediction in normalized_references
    return TaskScore(
        sample_id=target.sample_id,
        metric="exact_match",
        value=1.0 if correct else 0.0,
        correct=correct,
        task_type=target.task_type,
    )


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    mean_score: float
    accuracy: float
    per_task_mean: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_score": self.mean_score,
            "accuracy": self.accuracy,
            "per_task_mean": dict(self.per_task_mean),
        }


def aggregate_scores(scores: Iterable[TaskScore]) -> MetricSummary:
    items = tuple(scores)
    if not items:
        return MetricSummary(0, 0.0, 0.0, {})
    grouped: dict[str, list[float]] = {}
    for item in items:
        grouped.setdefault(item.task_type, []).append(float(item.value))
    return MetricSummary(
        count=len(items),
        mean_score=sum(item.value for item in items) / len(items),
        accuracy=sum(1 for item in items if item.correct) / len(items),
        per_task_mean={key: sum(values) / len(values) for key, values in grouped.items()},
    )
