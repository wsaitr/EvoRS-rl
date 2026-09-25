from __future__ import annotations

import json

from rs_hmcomm.contracts import CostVector
from rs_hmcomm.evolution import (
    Candidate,
    DeterministicFieldVariation,
    EvolutionBudget,
    EvolutionConfig,
    EvolutionTermination,
    FitnessEvaluation,
    IntraAgentEvolution,
    VariationContext,
    VariationOutcome,
)


class AnswerFitness:
    name = "answer_quality_fixture"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls = 0

    def evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        self.calls += 1
        answer = str(candidate.payload["answer"])
        return FitnessEvaluation(
            score=self.scores[answer],
            components={"quality": self.scores[answer]},
            cost=CostVector(model_calls=1, model_input_tokens=2),
        )


class IndexedVariation:
    name = "indexed_answer_variation"

    def vary(
        self,
        parent: Candidate,
        *,
        context: VariationContext,
    ) -> VariationOutcome:
        variants = ("weak-private-marker", "best-public-marker")
        return VariationOutcome(
            payload={"answer": variants[context.offspring_index % len(variants)]}
        )


def test_real_population_variation_fitness_selection_and_private_trace() -> None:
    evaluator = AnswerFitness(
        {
            "seed": 0.1,
            "weak-private-marker": 0.2,
            "best-public-marker": 0.9,
        }
    )
    engine = IntraAgentEvolution(
        IndexedVariation(),
        evaluator,
        EvolutionConfig(
            enabled=True,
            population_size=3,
            max_generations=1,
            no_progress_generations=2,
            budget=EvolutionBudget(
                max_candidate_evaluations=3,
                max_variations=2,
            ),
        ),
    )

    result = engine.run(Candidate("seed-candidate", {"answer": "seed"}))

    assert result.termination == EvolutionTermination.MAX_GENERATIONS
    assert result.selected_candidate.payload["answer"] == "best-public-marker"
    assert result.metrics.candidates_evaluated == 3
    assert result.metrics.variations_applied == 2
    assert result.metrics.extra_cost.model_calls == 3
    assert result.metrics.extra_cost.model_input_tokens == 6

    generation = result.private_trace.generations[0]
    assert len(generation.population) == 3
    assert len(generation.variations) == 2
    assert len(generation.fitness_evaluations) == 3
    assert generation.selection.selected_candidate_id == result.selected_candidate.candidate_id

    public_json = json.dumps(result.to_dict(), sort_keys=True)
    private_json = json.dumps(result.private_trace.to_dict(), sort_keys=True)
    assert "weak-private-marker" not in public_json
    assert "weak-private-marker" in private_json
    assert "best-public-marker" in public_json


class ExplodingOperator:
    name = "must_not_run"

    def vary(self, parent: Candidate, *, context: VariationContext) -> VariationOutcome:
        raise AssertionError("disabled evolution called variation")


class ExplodingEvaluator:
    name = "must_not_run"

    def evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        raise AssertionError("disabled evolution called fitness")


def test_disabled_evolution_is_a_zero_cost_baseline_passthrough() -> None:
    seed = Candidate("baseline-seed", {"answer": "unchanged"})
    result = IntraAgentEvolution(
        ExplodingOperator(),
        ExplodingEvaluator(),
        EvolutionConfig(
            enabled=False,
            population_size=1,
            max_generations=1,
            no_progress_generations=1,
            budget=EvolutionBudget(1, 0),
        ),
    ).run(seed)

    assert result.termination == EvolutionTermination.DISABLED
    assert result.selected_candidate is seed
    assert result.selected_fitness is None
    assert result.metrics.candidates_evaluated == 0
    assert result.metrics.extra_cost == CostVector()
    assert result.private_trace.generations == ()


def test_default_evolution_config_is_disabled() -> None:
    seed = Candidate("default-disabled-seed", {"answer": "unchanged"})
    result = IntraAgentEvolution(
        ExplodingOperator(),
        ExplodingEvaluator(),
        EvolutionConfig(),
    ).run(seed)

    assert result.termination == EvolutionTermination.DISABLED
    assert result.selected_candidate is seed
    assert result.metrics.extra_cost == CostVector()


class IncrementVariation:
    name = "increment_quality"

    def vary(self, parent: Candidate, *, context: VariationContext) -> VariationOutcome:
        return VariationOutcome(payload={"quality": int(parent.payload["quality"]) + 1})


class NumericFitness:
    name = "numeric_quality"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        self.calls += 1
        return FitnessEvaluation(score=float(candidate.payload["quality"]))


def test_hard_budget_stops_before_an_operation_would_exceed_it() -> None:
    evaluator = NumericFitness()
    config = EvolutionConfig(
        enabled=True,
        population_size=2,
        max_generations=5,
        no_progress_generations=5,
        budget=EvolutionBudget(
            max_candidate_evaluations=3,
            max_variations=2,
        ),
    )
    result = IntraAgentEvolution(IncrementVariation(), evaluator, config).run(
        Candidate("budget-seed", {"quality": 0})
    )

    assert result.termination == EvolutionTermination.BUDGET_EXHAUSTED
    assert result.metrics.generations_completed == 2
    assert result.metrics.candidates_evaluated == 3
    assert result.metrics.variations_applied == 2
    assert evaluator.calls == config.budget.max_candidate_evaluations


class NoChangeVariation:
    name = "no_change"

    def vary(self, parent: Candidate, *, context: VariationContext) -> VariationOutcome:
        return VariationOutcome(payload={"quality": parent.payload["quality"]})


def test_no_progress_patience_terminates_evolution() -> None:
    result = IntraAgentEvolution(
        NoChangeVariation(),
        NumericFitness(),
        EvolutionConfig(
            enabled=True,
            population_size=2,
            max_generations=20,
            no_progress_generations=2,
            budget=EvolutionBudget(
                max_candidate_evaluations=20,
                max_variations=19,
            ),
        ),
    ).run(Candidate("stagnant-seed", {"quality": 1}))

    assert result.termination == EvolutionTermination.NO_PROGRESS
    assert result.metrics.generations_completed == 2
    assert result.metrics.candidates_evaluated == 3
    assert result.metrics.variations_applied == 2


def test_seeded_field_variation_is_reproducible() -> None:
    config = EvolutionConfig(
        enabled=True,
        population_size=2,
        max_generations=2,
        no_progress_generations=2,
        seed=41,
        budget=EvolutionBudget(3, 2),
    )
    variation = DeterministicFieldVariation(
        field_name="answer",
        choices=("alpha", "beta", "gamma"),
    )
    seed = Candidate("repeatable-seed", {"answer": "seed"})

    first = IntraAgentEvolution(
        variation,
        AnswerFitness({"seed": 0.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        config,
    ).run(seed)
    second = IntraAgentEvolution(
        variation,
        AnswerFitness({"seed": 0.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        config,
    ).run(seed)

    assert first.to_dict() == second.to_dict()
    assert first.private_trace.to_dict() == second.private_trace.to_dict()
