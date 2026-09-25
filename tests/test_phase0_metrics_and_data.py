from __future__ import annotations

import pytest

from rs_hmcomm.contracts.experiment import (
    DatasetSpec,
    canonical_manifest_digest,
    deterministic_sample_ids,
)
from rs_hmcomm.contracts.metrics import (
    EvaluationTarget,
    TaskScore,
    aggregate_scores,
    exact_match_score,
)


def test_dataset_spec_and_manifest_digest_are_reproducible() -> None:
    records = [
        {"sample_id": "a", "image_ref": "images/a.png"},
        {"sample_id": "b", "image_ref": "images/b.png"},
    ]
    digest = canonical_manifest_digest(records)
    spec = DatasetSpec("fixture", "v1", "test", seed=7, limit=2, manifest_digest=digest)

    assert DatasetSpec.from_dict(spec.to_dict()) == spec
    assert canonical_manifest_digest(records) == digest
    assert canonical_manifest_digest(reversed(records)) != digest


def test_sample_selection_is_seeded_and_input_order_independent() -> None:
    ids = ["sample-1", "sample-2", "sample-3", "sample-4"]
    first = deterministic_sample_ids(ids, seed=20260820, limit=3)
    second = deterministic_sample_ids(reversed(ids), seed=20260820, limit=3)
    different = deterministic_sample_ids(ids, seed=20260821, limit=3)

    assert first == second
    assert first != different
    assert len(first) == 3


def test_sample_selection_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        deterministic_sample_ids(["a", "a"], seed=1)


def test_exact_match_labels_stay_outside_agent_task_state() -> None:
    target = EvaluationTarget("sample-1", ("Six aircraft", "6 aircraft"), "counting")
    score = exact_match_score("  SIX   AIRCRAFT! ", target)

    assert score.correct
    assert score.value == 1.0
    assert EvaluationTarget.from_dict(target.to_dict()) == target
    assert TaskScore.from_dict(score.to_dict()) == score


def test_metric_aggregation_is_deterministic() -> None:
    scores = [
        TaskScore("a", "exact_match", 1.0, True, "counting"),
        TaskScore("b", "exact_match", 0.0, False, "counting"),
        TaskScore("c", "exact_match", 1.0, True, "classification"),
    ]
    summary = aggregate_scores(scores)
    assert summary.count == 3
    assert summary.mean_score == pytest.approx(2 / 3)
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.per_task_mean == {"counting": 0.5, "classification": 1.0}


def test_metric_deserialization_fails_closed() -> None:
    with pytest.raises(ValueError, match="boolean"):
        TaskScore.from_dict(
            {
                "sample_id": "a",
                "metric": "exact_match",
                "value": 0.0,
                "correct": "false",
                "task_type": "vqa",
            }
        )
    with pytest.raises(ValueError, match="unknown fields"):
        EvaluationTarget.from_dict(
            {
                "sample_id": "a",
                "references": ["x"],
                "task_type": "vqa",
                "unexpected": True,
            }
        )
