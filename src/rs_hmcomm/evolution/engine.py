"""A minimal, deterministic (mu + lambda) intra-Agent evolution engine."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable

from rs_hmcomm.contracts import CostVector
from rs_hmcomm.contracts._validation import json_object_copy, require_text

from .protocol import (
    Candidate,
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


@dataclass(frozen=True, slots=True)
class DeterministicFieldVariation:
    """Replace one payload field with a seeded choice different from its value."""

    field_name: str
    choices: tuple[Any, ...]
    name: str = "deterministic_field_choice"

    def __post_init__(self) -> None:
        require_text(self.field_name, "field_name")
        require_text(self.name, "variation name")
        object.__setattr__(self, "choices", tuple(self.choices))
        if not self.choices:
            raise ValueError("choices must contain at least one value")
        # Candidate validation is the single source of JSON-boundary checks.
        Candidate("choice-validation", {"value": list(self.choices)})

    def vary(
        self,
        parent: Candidate,
        *,
        context: VariationContext,
    ) -> VariationOutcome:
        payload = json_object_copy(parent.payload)
        current = payload.get(self.field_name)
        alternatives = [item for item in self.choices if item != current]
        if not alternatives:
            raise ValueError("variation choices do not contain an alternative value")
        payload[self.field_name] = alternatives[context.rng.randrange(len(alternatives))]
        return VariationOutcome(payload=payload)


class IntraAgentEvolution:
    """Evolve private candidates and expose only the deterministic winner.

    This is a ``(1 + lambda)`` strategy.  The incumbent is retained while
    ``population_size - 1`` children are varied and scored each generation.
    Fitness is maximized.  Ties are resolved by candidate ID so identical
    seeds and operators produce identical selections.
    """

    selection_strategy = "max_fitness_then_candidate_id"

    def __init__(
        self,
        variation: VariationOperator,
        evaluator: FitnessEvaluator,
        config: EvolutionConfig | None = None,
    ) -> None:
        self.variation = variation
        self.evaluator = evaluator
        self.config = config or EvolutionConfig()
        require_text(self.variation.name, "variation name")
        require_text(self.evaluator.name, "fitness evaluator name")

    def run(
        self,
        seed_candidate: Candidate,
        *,
        event_hook: Callable[[str, object], None] | None = None,
    ) -> EvolutionRunResult:
        if not isinstance(seed_candidate, Candidate):
            raise TypeError("seed_candidate must be a Candidate")
        if not self.config.enabled:
            return EvolutionRunResult(
                selected_candidate=seed_candidate,
                selected_fitness=None,
                termination=EvolutionTermination.DISABLED,
                metrics=EvolutionMetrics(
                    population_size=1,
                    generations_completed=0,
                    candidates_evaluated=0,
                    variations_applied=0,
                    initial_fitness=None,
                    selected_fitness=None,
                    fitness_gain=0.0,
                ),
                _private_trace=PrivateEvolutionTrace(),
            )

        rng = random.Random(self.config.seed)
        incumbent = seed_candidate
        incumbent_fitness: FitnessEvaluation | None = None
        initial_fitness: float | None = None
        evaluations = 0
        variations = 0
        no_progress = 0
        extra_cost = CostVector()
        generations: list[GenerationRecord] = []
        termination = EvolutionTermination.MAX_GENERATIONS

        def notify(event_name: str, record: object) -> None:
            if event_hook is not None:
                event_hook(event_name, record)

        notify("candidate_proposed", incumbent)

        for generation in range(1, self.config.max_generations + 1):
            evaluation_calls_needed = (
                self.config.population_size
                if incumbent_fitness is None
                else self.config.population_size - 1
            )
            variation_calls_needed = self.config.population_size - 1
            budget = self.config.budget
            if (
                evaluations + evaluation_calls_needed
                > budget.max_candidate_evaluations
                or variations + variation_calls_needed > budget.max_variations
            ):
                termination = EvolutionTermination.BUDGET_EXHAUSTED
                break

            population: list[Candidate] = [incumbent]
            variation_records: list[VariationRecord] = []
            fitness_records: list[FitnessRecord] = []

            if incumbent_fitness is None:
                incumbent_fitness = self._evaluate(incumbent)
                evaluations += 1
                extra_cost = extra_cost + incumbent_fitness.cost
                initial_fitness = incumbent_fitness.score
                incumbent_record = FitnessRecord(
                    candidate_id=incumbent.candidate_id,
                    evaluator=self.evaluator.name,
                    evaluation=incumbent_fitness,
                )
                fitness_records.append(incumbent_record)
                notify("candidate_evaluated", incumbent_record)

            fitness_by_id: dict[str, FitnessEvaluation] = {
                incumbent.candidate_id: incumbent_fitness
            }
            for offspring_index in range(self.config.population_size - 1):
                context = VariationContext(
                    generation=generation,
                    offspring_index=offspring_index,
                    rng=rng,
                )
                outcome = self.variation.vary(incumbent, context=context)
                if not isinstance(outcome, VariationOutcome):
                    raise TypeError("variation operator must return VariationOutcome")
                child_id = (
                    f"{seed_candidate.candidate_id}-g{generation}-v{offspring_index}"
                )
                child = Candidate(
                    candidate_id=child_id,
                    payload=outcome.payload,
                    generation=generation,
                    parent_id=incumbent.candidate_id,
                    variation_operator=self.variation.name,
                )
                notify("candidate_proposed", child)
                variations += 1
                extra_cost = extra_cost + outcome.cost
                population.append(child)
                variation_records.append(
                    VariationRecord(
                        variation_id=f"variation-g{generation}-v{offspring_index}",
                        operator=self.variation.name,
                        parent_candidate_id=incumbent.candidate_id,
                        child_candidate_id=child.candidate_id,
                        generation=generation,
                        cost=outcome.cost,
                    )
                )

                child_fitness = self._evaluate(child)
                evaluations += 1
                extra_cost = extra_cost + child_fitness.cost
                fitness_by_id[child.candidate_id] = child_fitness
                child_record = FitnessRecord(
                    candidate_id=child.candidate_id,
                    evaluator=self.evaluator.name,
                    evaluation=child_fitness,
                )
                fitness_records.append(child_record)
                notify("candidate_evaluated", child_record)

            previous_score = incumbent_fitness.score
            selected = min(
                population,
                key=lambda item: (
                    -fitness_by_id[item.candidate_id].score,
                    item.candidate_id,
                ),
            )
            selected_fitness = fitness_by_id[selected.candidate_id]
            fitness_gain = max(0.0, selected_fitness.score - previous_score)
            selection = SelectionRecord(
                strategy=self.selection_strategy,
                candidate_scores={
                    item.candidate_id: fitness_by_id[item.candidate_id].score
                    for item in population
                },
                selected_candidate_id=selected.candidate_id,
            )
            generation_record = GenerationRecord(
                generation=generation,
                population=tuple(population),
                variations=tuple(variation_records),
                fitness_evaluations=tuple(fitness_records),
                selection=selection,
                fitness_gain=fitness_gain,
            )
            generations.append(generation_record)
            notify("candidate_selected", generation_record)
            incumbent = selected
            incumbent_fitness = selected_fitness

            if fitness_gain > self.config.min_fitness_gain:
                no_progress = 0
            else:
                no_progress += 1
                if no_progress >= self.config.no_progress_generations:
                    termination = EvolutionTermination.NO_PROGRESS
                    break

        if incumbent_fitness is None or initial_fitness is None:
            # Enabled configs are validated to fit generation one, so reaching
            # this branch would indicate a broken internal accounting rule.
            raise RuntimeError("evolution terminated before the seed was evaluated")

        return EvolutionRunResult(
            selected_candidate=incumbent,
            selected_fitness=incumbent_fitness,
            termination=termination,
            metrics=EvolutionMetrics(
                population_size=self.config.population_size,
                generations_completed=len(generations),
                candidates_evaluated=evaluations,
                variations_applied=variations,
                initial_fitness=initial_fitness,
                selected_fitness=incumbent_fitness.score,
                fitness_gain=max(0.0, incumbent_fitness.score - initial_fitness),
                extra_cost=extra_cost,
            ),
            _private_trace=PrivateEvolutionTrace(tuple(generations)),
        )

    def _evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        evaluation = self.evaluator.evaluate(candidate)
        if not isinstance(evaluation, FitnessEvaluation):
            raise TypeError("fitness evaluator must return FitnessEvaluation")
        return evaluation


__all__ = ["DeterministicFieldVariation", "IntraAgentEvolution"]
