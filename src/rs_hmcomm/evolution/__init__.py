"""Single-Agent internal evolution primitives."""

from .engine import DeterministicFieldVariation, IntraAgentEvolution
from .protocol import (
    Candidate,
    EvolutionBudget,
    EvolutionConfig,
    EvolutionMetrics,
    EvolutionRunResult,
    EvolutionTermination,
    FitnessEvaluation,
    FitnessEvaluator,
    FitnessRecord,
    GenerationRecord,
    PrivateEvolutionTrace,
    SelectionRecord,
    VariationContext,
    VariationOperator,
    VariationOutcome,
    VariationRecord,
)

__all__ = [
    "Candidate",
    "DeterministicFieldVariation",
    "EvolutionBudget",
    "EvolutionConfig",
    "EvolutionMetrics",
    "EvolutionRunResult",
    "EvolutionTermination",
    "FitnessEvaluation",
    "FitnessEvaluator",
    "FitnessRecord",
    "GenerationRecord",
    "IntraAgentEvolution",
    "PrivateEvolutionTrace",
    "SelectionRecord",
    "VariationContext",
    "VariationOperator",
    "VariationOutcome",
    "VariationRecord",
]
