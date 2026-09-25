"""Contracts for deterministic, intra-Agent candidate evolution.

These records deliberately live outside the shared Agent contracts.  A
population is private scratch state: only the selected candidate is allowed to
cross the Agent boundary.  The full trace is available for research logging,
but must be requested explicitly through ``EvolutionRunResult.private_trace``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import random
from numbers import Real
from typing import Any, Mapping, Protocol

from rs_hmcomm.contracts import CostVector
from rs_hmcomm.contracts._validation import (
    json_object_copy,
    normalize_json_object,
    require_id,
    require_text,
)


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class Candidate:
    """One private candidate genome owned by a single Agent invocation."""

    candidate_id: str
    payload: Mapping[str, Any]
    generation: int = 0
    parent_id: str | None = None
    variation_operator: str = "seed"

    def __post_init__(self) -> None:
        require_id(self.candidate_id, "candidate_id")
        _non_negative_int(self.generation, "generation")
        if self.parent_id is not None:
            require_id(self.parent_id, "parent_id")
            if self.parent_id == self.candidate_id:
                raise ValueError("a candidate cannot be its own parent")
        require_id(self.variation_operator, "variation_operator")
        object.__setattr__(
            self,
            "payload",
            normalize_json_object(self.payload, "candidate payload"),
        )

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "payload": json_object_copy(self.payload),
            "generation": self.generation,
            "parent_id": self.parent_id,
            "variation_operator": self.variation_operator,
        }

    def to_selected_dict(self) -> dict[str, Any]:
        """Serialize only fields needed to commit the selected result."""

        return {
            "candidate_id": self.candidate_id,
            "payload": json_object_copy(self.payload),
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class VariationContext:
    """Deterministic context supplied to a variation operator."""

    generation: int
    offspring_index: int
    rng: random.Random = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_int(self.generation, "generation")
        _non_negative_int(self.offspring_index, "offspring_index")
        if not isinstance(self.rng, random.Random):
            raise TypeError("rng must be an instance of random.Random")


@dataclass(frozen=True, slots=True)
class VariationOutcome:
    """A mutated genome plus the extra cost paid to produce it."""

    payload: Mapping[str, Any]
    cost: CostVector = field(default_factory=CostVector)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            normalize_json_object(self.payload, "variation payload"),
        )
        if not isinstance(self.cost, CostVector):
            raise TypeError("variation cost must be a CostVector")


class VariationOperator(Protocol):
    """Domain-specific mutation; it may not choose IDs or commit records."""

    name: str

    def vary(
        self,
        parent: Candidate,
        *,
        context: VariationContext,
    ) -> VariationOutcome:
        ...


@dataclass(frozen=True, slots=True)
class FitnessEvaluation:
    """Scalar maximization fitness and its measured evaluation cost."""

    score: float
    components: Mapping[str, float] = field(default_factory=dict)
    cost: CostVector = field(default_factory=CostVector)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _finite_float(self.score, "fitness score"))
        if not isinstance(self.components, Mapping):
            raise TypeError("fitness components must be a mapping")
        normalized_components: dict[str, float] = {}
        for key, value in self.components.items():
            require_text(key, "fitness component name")
            normalized_components[key] = _finite_float(
                value,
                f"fitness component {key}",
            )
        object.__setattr__(
            self,
            "components",
            normalize_json_object(normalized_components, "fitness components"),
        )
        if not isinstance(self.cost, CostVector):
            raise TypeError("fitness cost must be a CostVector")


class FitnessEvaluator(Protocol):
    """Scores one candidate without observing any evaluation label by default."""

    name: str

    def evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        ...


@dataclass(frozen=True, slots=True)
class EvolutionBudget:
    """Hard discrete limits checked before any variation or evaluation call."""

    max_candidate_evaluations: int
    max_variations: int

    def __post_init__(self) -> None:
        _positive_int(self.max_candidate_evaluations, "max_candidate_evaluations")
        _non_negative_int(self.max_variations, "max_variations")


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    enabled: bool = False
    population_size: int = 2
    max_generations: int = 2
    no_progress_generations: int = 2
    min_fitness_gain: float = 0.0
    seed: int = 0
    budget: EvolutionBudget = field(
        default_factory=lambda: EvolutionBudget(
            max_candidate_evaluations=3,
            max_variations=2,
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        _positive_int(self.population_size, "population_size")
        _positive_int(self.max_generations, "max_generations")
        _positive_int(self.no_progress_generations, "no_progress_generations")
        gain = _finite_float(self.min_fitness_gain, "min_fitness_gain")
        if gain < 0:
            raise ValueError("min_fitness_gain must be non-negative")
        object.__setattr__(self, "min_fitness_gain", gain)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.budget, EvolutionBudget):
            raise TypeError("budget must be an EvolutionBudget")
        if self.enabled:
            if self.population_size < 2:
                raise ValueError("enabled evolution requires at least two candidates")
            if self.budget.max_candidate_evaluations < self.population_size:
                raise ValueError(
                    "evaluation budget must fit the first complete population"
                )
            if self.budget.max_variations < self.population_size - 1:
                raise ValueError(
                    "variation budget must fit the first complete population"
                )


class EvolutionTermination(str, Enum):
    DISABLED = "disabled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    MAX_GENERATIONS = "max_generations"


@dataclass(frozen=True, slots=True)
class VariationRecord:
    variation_id: str
    operator: str
    parent_candidate_id: str
    child_candidate_id: str
    generation: int
    cost: CostVector = field(default_factory=CostVector)

    def __post_init__(self) -> None:
        require_id(self.variation_id, "variation_id")
        require_id(self.operator, "operator")
        require_id(self.parent_candidate_id, "parent_candidate_id")
        require_id(self.child_candidate_id, "child_candidate_id")
        _positive_int(self.generation, "generation")
        if not isinstance(self.cost, CostVector):
            raise TypeError("variation record cost must be a CostVector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variation_id": self.variation_id,
            "operator": self.operator,
            "parent_candidate_id": self.parent_candidate_id,
            "child_candidate_id": self.child_candidate_id,
            "generation": self.generation,
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FitnessRecord:
    candidate_id: str
    evaluator: str
    evaluation: FitnessEvaluation

    def __post_init__(self) -> None:
        require_id(self.candidate_id, "candidate_id")
        require_id(self.evaluator, "evaluator")
        if not isinstance(self.evaluation, FitnessEvaluation):
            raise TypeError("evaluation must be a FitnessEvaluation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evaluator": self.evaluator,
            "score": self.evaluation.score,
            "components": json_object_copy(self.evaluation.components),
            "cost": self.evaluation.cost.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    strategy: str
    candidate_scores: Mapping[str, float]
    selected_candidate_id: str

    def __post_init__(self) -> None:
        require_id(self.strategy, "selection strategy")
        require_id(self.selected_candidate_id, "selected_candidate_id")
        if not isinstance(self.candidate_scores, Mapping) or len(self.candidate_scores) < 2:
            raise ValueError("selection requires scores for at least two candidates")
        scores: dict[str, float] = {}
        for candidate_id, score in self.candidate_scores.items():
            require_id(candidate_id, "selection candidate_id")
            scores[candidate_id] = _finite_float(score, "selection score")
        if self.selected_candidate_id not in scores:
            raise ValueError("selected_candidate_id is not in candidate_scores")
        object.__setattr__(
            self,
            "candidate_scores",
            normalize_json_object(scores, "candidate_scores"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "candidate_scores": json_object_copy(self.candidate_scores),
            "selected_candidate_id": self.selected_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    population: tuple[Candidate, ...]
    variations: tuple[VariationRecord, ...]
    fitness_evaluations: tuple[FitnessRecord, ...]
    selection: SelectionRecord
    fitness_gain: float

    def __post_init__(self) -> None:
        _positive_int(self.generation, "generation")
        object.__setattr__(self, "population", tuple(self.population))
        object.__setattr__(self, "variations", tuple(self.variations))
        object.__setattr__(self, "fitness_evaluations", tuple(self.fitness_evaluations))
        if len(self.population) < 2:
            raise ValueError("a generation requires at least two candidates")
        candidate_ids = [item.candidate_id for item in self.population]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("generation population contains duplicate candidate IDs")
        if not isinstance(self.selection, SelectionRecord):
            raise TypeError("selection must be a SelectionRecord")
        if set(self.selection.candidate_scores) != set(candidate_ids):
            raise ValueError("selection scores must cover the complete population")
        gain = _finite_float(self.fitness_gain, "fitness_gain")
        if gain < 0:
            raise ValueError("fitness_gain cannot be negative")
        object.__setattr__(self, "fitness_gain", gain)

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "population": [item.to_private_dict() for item in self.population],
            "variations": [item.to_dict() for item in self.variations],
            "fitness_evaluations": [item.to_dict() for item in self.fitness_evaluations],
            "selection": self.selection.to_dict(),
            "fitness_gain": self.fitness_gain,
        }


@dataclass(frozen=True, slots=True)
class PrivateEvolutionTrace:
    """Private scratch trace; never include this object in an AgentResult."""

    generations: tuple[GenerationRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "generations", tuple(self.generations))
        expected = tuple(range(1, len(self.generations) + 1))
        actual = tuple(item.generation for item in self.generations)
        if actual != expected:
            raise ValueError("private evolution generations must be contiguous from one")

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(
            candidate.candidate_id
            for generation in self.generations
            for candidate in generation.population
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "private_agent_scratchpad",
            "generations": [item.to_private_dict() for item in self.generations],
        }


@dataclass(frozen=True, slots=True)
class EvolutionMetrics:
    population_size: int
    generations_completed: int
    candidates_evaluated: int
    variations_applied: int
    initial_fitness: float | None
    selected_fitness: float | None
    fitness_gain: float
    extra_cost: CostVector = field(default_factory=CostVector)

    def __post_init__(self) -> None:
        _positive_int(self.population_size, "population_size")
        _non_negative_int(self.generations_completed, "generations_completed")
        _non_negative_int(self.candidates_evaluated, "candidates_evaluated")
        _non_negative_int(self.variations_applied, "variations_applied")
        for field_name in ("initial_fitness", "selected_fitness"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_float(value, field_name))
        gain = _finite_float(self.fitness_gain, "fitness_gain")
        if gain < 0:
            raise ValueError("fitness_gain cannot be negative")
        object.__setattr__(self, "fitness_gain", gain)
        if not isinstance(self.extra_cost, CostVector):
            raise TypeError("extra_cost must be a CostVector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "generations_completed": self.generations_completed,
            "candidates_evaluated": self.candidates_evaluated,
            "variations_applied": self.variations_applied,
            "initial_fitness": self.initial_fitness,
            "selected_fitness": self.selected_fitness,
            "fitness_gain": self.fitness_gain,
            "extra_cost": self.extra_cost.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvolutionRunResult:
    """Public selection plus an explicitly private research trace."""

    selected_candidate: Candidate
    selected_fitness: FitnessEvaluation | None
    termination: EvolutionTermination
    metrics: EvolutionMetrics
    _private_trace: PrivateEvolutionTrace = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selected_candidate, Candidate):
            raise TypeError("selected_candidate must be a Candidate")
        if self.selected_fitness is not None and not isinstance(
            self.selected_fitness,
            FitnessEvaluation,
        ):
            raise TypeError("selected_fitness must be a FitnessEvaluation or None")
        if not isinstance(self.termination, EvolutionTermination):
            raise TypeError("termination must be an EvolutionTermination")
        if not isinstance(self.metrics, EvolutionMetrics):
            raise TypeError("metrics must be EvolutionMetrics")
        if not isinstance(self._private_trace, PrivateEvolutionTrace):
            raise TypeError("_private_trace must be PrivateEvolutionTrace")

    @property
    def private_trace(self) -> PrivateEvolutionTrace:
        """Return private scratch data for local logging, never workspace commit."""

        return self._private_trace

    def to_dict(self) -> dict[str, Any]:
        """Serialize the public boundary; discarded candidates are omitted."""

        return {
            "selected_candidate": self.selected_candidate.to_selected_dict(),
            "selected_fitness": (
                self.selected_fitness.score
                if self.selected_fitness is not None
                else None
            ),
            "termination": self.termination.value,
            "metrics": self.metrics.to_dict(),
        }


__all__ = [
    "Candidate",
    "EvolutionBudget",
    "EvolutionConfig",
    "EvolutionMetrics",
    "EvolutionRunResult",
    "EvolutionTermination",
    "FitnessEvaluation",
    "FitnessEvaluator",
    "FitnessRecord",
    "GenerationRecord",
    "PrivateEvolutionTrace",
    "SelectionRecord",
    "VariationContext",
    "VariationOperator",
    "VariationOutcome",
    "VariationRecord",
]
