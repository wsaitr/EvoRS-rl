from __future__ import annotations

import json

import pytest

from rs_hmcomm.contracts import (
    AgentObservation,
    AgentResult,
    AgentResultStatus,
    AnswerCandidate,
    Assignment,
    BudgetExceeded,
    BudgetLimits,
    Claim,
    CostVector,
    Evidence,
    TaskContext,
)


def observation() -> AgentObservation:
    return AgentObservation(
        observation_id="observation-0",
        agent_id="generalist",
        task=TaskContext(
            task_id="task-1",
            image_ref="fixture://airport.png",
            question="How many aircraft are visible?",
            task_type="counting",
        ),
        assignment=Assignment(
            assignment_id="assignment-1",
            agent_id="generalist",
            objective="Answer with visual evidence",
        ),
        step=0,
        budget_limits=BudgetLimits(max_agent_calls=3, max_tool_calls=2),
    )


def valid_result() -> AgentResult:
    evidence = Evidence(
        evidence_id="evidence-1",
        kind="crop",
        summary="Aircraft are visible in the selected ROI",
        created_by="generalist",
        artifact_refs=("artifact-crop-1",),
        confidence=0.8,
    )
    claim = Claim(
        claim_id="claim-1",
        statement="There are six aircraft",
        created_by="generalist",
        evidence_ids=(evidence.evidence_id,),
        confidence=0.75,
    )
    answer = AnswerCandidate(
        candidate_id="answer-1",
        answer="6",
        created_by="generalist",
        claim_ids=(claim.claim_id,),
        evidence_ids=(evidence.evidence_id,),
        confidence=0.75,
    )
    return AgentResult(
        result_id="result-1",
        agent_id="generalist",
        assignment_id="assignment-1",
        status=AgentResultStatus.COMPLETED,
        evidence=(evidence,),
        claims=(claim,),
        answer_candidates=(answer,),
    )


def test_agent_boundary_round_trips_and_validates_references() -> None:
    obs = observation()
    result = valid_result()

    result.validate_against(obs)
    decoded_obs = AgentObservation.from_dict(json.loads(json.dumps(obs.to_dict())))
    decoded_result = AgentResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert decoded_obs == obs
    assert decoded_result == result
    decoded_result.validate_against(decoded_obs)


def test_agent_result_rejects_dangling_evidence_reference() -> None:
    result = valid_result()
    dangling_claim = Claim(
        claim_id="claim-dangling",
        statement="Unsupported",
        created_by="generalist",
        evidence_ids=("missing-evidence",),
    )
    invalid = AgentResult(
        result_id="result-2",
        agent_id="generalist",
        assignment_id="assignment-1",
        status="completed",
        claims=(dangling_claim,),
    )

    with pytest.raises(ValueError, match="dangling evidence"):
        invalid.validate_against(observation())


def test_agent_result_cannot_overwrite_visible_record() -> None:
    existing = valid_result().evidence[0]
    obs = AgentObservation(
        **{
            **observation().__dict__,
            "evidence": (existing,),
        }
    )
    duplicate = AgentResult(
        result_id="result-duplicate",
        agent_id="generalist",
        assignment_id="assignment-1",
        status="completed",
        evidence=(existing,),
    )
    with pytest.raises(ValueError, match="replace visible"):
        duplicate.validate_against(obs)


def test_records_reject_runtime_objects_and_invalid_confidence() -> None:
    with pytest.raises(TypeError, match="unsupported value"):
        TaskContext(
            task_id="task-1",
            image_ref="fixture://image.png",
            question="Question?",
            metadata={"image": object()},
        )
    with pytest.raises(ValueError, match="within"):
        Evidence(
            evidence_id="evidence-1",
            kind="visual",
            summary="summary",
            created_by="generalist",
            confidence=1.1,
        )


def test_observation_rejects_assignment_for_other_agent() -> None:
    obs = observation()
    with pytest.raises(ValueError, match="assignment.agent_id"):
        AgentObservation(
            observation_id=obs.observation_id,
            agent_id="visual",
            task=obs.task,
            assignment=obs.assignment,
            step=obs.step,
            budget_limits=obs.budget_limits,
        )


def test_observation_rejects_cross_type_id_collision() -> None:
    evidence = Evidence(
        evidence_id="shared-id",
        kind="visual",
        summary="visible object",
        created_by="generalist",
    )
    claim = Claim(
        claim_id="shared-id",
        statement="claim",
        created_by="generalist",
        evidence_ids=(evidence.evidence_id,),
    )
    base = observation()
    with pytest.raises(ValueError, match="global namespace"):
        AgentObservation(
            observation_id=base.observation_id,
            agent_id=base.agent_id,
            task=base.task,
            assignment=base.assignment,
            step=base.step,
            budget_limits=base.budget_limits,
            evidence=(evidence,),
            claims=(claim,),
        )


def test_observation_rejects_dangling_scoped_refs() -> None:
    base = observation()
    dangling = Assignment(
        assignment_id="assignment-2",
        agent_id="generalist",
        objective="Use scoped context",
        context_refs=("missing-record",),
    )
    with pytest.raises(ValueError, match="dangling context"):
        AgentObservation(
            observation_id="observation-2",
            agent_id="generalist",
            task=base.task,
            assignment=dangling,
            step=0,
            budget_limits=base.budget_limits,
        )


def test_result_rejects_capability_request_self_reference() -> None:
    from rs_hmcomm.contracts import CapabilityRequest

    request = CapabilityRequest(
        request_id="request-1",
        capability="knowledge",
        objective="Need domain knowledge",
        created_by="generalist",
        context_refs=("request-1",),
    )
    result = AgentResult(
        result_id="result-request",
        agent_id="generalist",
        assignment_id="assignment-1",
        status="needs_capability",
        capability_requests=(request,),
    )
    with pytest.raises(ValueError, match="dangling context"):
        result.validate_against(observation())


def test_result_validation_applies_measured_runtime_cost_to_budget() -> None:
    obs = AgentObservation(
        observation_id="observation-budget",
        agent_id="generalist",
        task=observation().task,
        assignment=observation().assignment,
        step=0,
        budget_limits=BudgetLimits(max_model_calls=0),
    )
    with pytest.raises(BudgetExceeded, match="budget exceeded"):
        valid_result().validate_against(obs, observed_cost_delta=CostVector(model_calls=1))


def test_observation_rejects_cost_already_over_budget() -> None:
    base = observation()
    with pytest.raises(BudgetExceeded, match="budget exceeded"):
        AgentObservation(
            observation_id="observation-over-budget",
            agent_id=base.agent_id,
            task=base.task,
            assignment=base.assignment,
            step=base.step,
            budget_limits=BudgetLimits(max_model_calls=0),
            cost_so_far=CostVector(model_calls=1),
        )


def test_contract_metadata_is_deeply_immutable() -> None:
    task = TaskContext(
        task_id="task-immutable",
        image_ref="fixture://image.png",
        question="Question?",
        metadata={"nested": [1, 2]},
    )
    with pytest.raises(TypeError):
        task.metadata["new"] = 3
    assert isinstance(task.metadata["nested"], tuple)
