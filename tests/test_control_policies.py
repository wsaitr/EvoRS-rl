from __future__ import annotations

import pytest

from rs_hmcomm.control import (
    AbortDecision,
    AllAgentPolicy,
    ControllerObservation,
    DispatchDecision,
    EvoRSController,
    FinishDecision,
    FixedWorkflowPolicy,
    RuleRoutingPolicy,
)
from rs_hmcomm.contracts import (
    AnswerCandidate,
    Evidence,
    RecordStatus,
    StopReason,
    TaskContext,
)
from rs_hmcomm.workspace import WorkspaceState


def task(question: str) -> TaskContext:
    return TaskContext(
        task_id="task-control",
        image_ref="fixture://airport",
        question=question,
        answer_format="short_text",
    )


def workspace(
    current_task: TaskContext,
    *,
    candidate_status: RecordStatus | None = None,
    candidate_id: str = "answer-verified",
) -> WorkspaceState:
    if candidate_status is None:
        return WorkspaceState.initial("episode-control", current_task)
    evidence = Evidence(
        "evidence-control",
        "visual",
        "Evidence produced by an expert",
        "visual",
    )
    candidate = AnswerCandidate(
        candidate_id,
        "candidate answer",
        "reasoning",
        evidence_ids=(evidence.evidence_id,),
        status=candidate_status,
    )
    return WorkspaceState(
        episode_id="episode-control",
        version=1,
        task=current_task,
        evidence=(evidence,),
        answer_candidates=(candidate,),
    )


def observation(
    question: str,
    *,
    path: tuple[str, ...] = (),
    step: int | None = None,
    max_steps: int = 8,
    candidate_status: RecordStatus | None = None,
) -> ControllerObservation:
    current_task = task(question)
    return ControllerObservation(
        task=current_task,
        workspace=workspace(current_task, candidate_status=candidate_status),
        agent_path=path,
        step=len(path) if step is None else step,
        max_steps=max_steps,
    )


def assert_route(
    policy: object,
    question: str,
    expected: tuple[str, ...],
) -> None:
    controller = EvoRSController(policy)  # type: ignore[arg-type]
    path: tuple[str, ...] = ()
    for expert_id in expected:
        decision = controller.decide(observation(question, path=path))
        assert isinstance(decision, DispatchDecision)
        assert decision.expert_id == expert_id
        assert decision.objective.strip()
        path += (expert_id,)

    no_candidate = controller.decide(observation(question, path=path))
    assert no_candidate == AbortDecision(StopReason.NO_PROGRESS)
    finish = controller.decide(
        observation(
            question,
            path=path,
            candidate_status=RecordStatus.VERIFIED,
        )
    )
    assert finish == FinishDecision("answer-verified")


@pytest.mark.parametrize(
    "question",
    [
        "Count the aircraft in the image.",
        "How many aircraft are visible?",
        "统计图中的飞机。",
        "图中有多少架飞机？",
        "机场里有几架飞机？",
    ],
)
def test_rule_routing_counting_never_dispatches_knowledge(question: str) -> None:
    expected = ("visual", "reasoning", "verification")
    assert_route(RuleRoutingPolicy(), question, expected)
    assert "knowledge" not in expected


@pytest.mark.parametrize(
    "question",
    [
        "Classify the airport.",
        "What airport type is shown?",
        "Determine the airport classification.",
        "判断机场类型。",
    ],
)
def test_rule_routing_classification_adds_knowledge(question: str) -> None:
    assert_route(
        RuleRoutingPolicy(),
        question,
        ("visual", "knowledge", "reasoning", "verification"),
    )


def test_fixed_and_all_agent_policies_have_distinct_deterministic_orders() -> None:
    assert_route(
        FixedWorkflowPolicy(),
        "Describe the scene.",
        ("visual", "reasoning", "verification"),
    )
    assert_route(
        AllAgentPolicy(),
        "Describe the scene.",
        ("visual", "knowledge", "reasoning", "verification"),
    )


@pytest.mark.parametrize(
    "policy",
    [RuleRoutingPolicy(), FixedWorkflowPolicy(), AllAgentPolicy()],
)
def test_policies_abort_max_steps_before_dispatch_or_finish(policy: object) -> None:
    decision = EvoRSController(policy).decide(  # type: ignore[arg-type]
        observation(
            "How many aircraft?",
            path=("visual",),
            step=2,
            max_steps=2,
            candidate_status=RecordStatus.VERIFIED,
        )
    )
    assert decision == AbortDecision(StopReason.MAX_STEPS)


def test_unverified_candidate_cannot_be_finished_by_builtin_policy() -> None:
    decision = EvoRSController(FixedWorkflowPolicy()).decide(
        observation(
            "Describe the scene.",
            path=("visual", "reasoning", "verification"),
            candidate_status=RecordStatus.PROPOSED,
        )
    )
    assert decision == AbortDecision(StopReason.NO_PROGRESS)


class SpyPolicy:
    def __init__(self, decision: object) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, observation: ControllerObservation) -> object:
        self.calls += 1
        return self.decision


def test_controller_calls_policy_once_and_does_not_execute_expert() -> None:
    policy = SpyPolicy(DispatchDecision("visual", "Inspect the image"))
    controller = EvoRSController(policy)  # type: ignore[arg-type]

    assert controller.decide(observation("Question?")) == policy.decision
    assert policy.calls == 1


def test_controller_rejects_unknown_expert_and_dangling_finish_ref() -> None:
    with pytest.raises(ValueError, match="unknown expert_id"):
        EvoRSController(SpyPolicy(DispatchDecision("rogue", "Do work"))).decide(  # type: ignore[arg-type]
            observation("Question?")
        )

    with pytest.raises(ValueError, match="missing answer candidate"):
        EvoRSController(SpyPolicy(FinishDecision("missing-answer"))).decide(  # type: ignore[arg-type]
            observation("Question?")
        )


def test_controller_requires_explicit_max_steps_abort_at_limit() -> None:
    controller = EvoRSController(
        SpyPolicy(DispatchDecision("visual", "Inspect"))  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must abort with max_steps"):
        controller.decide(observation("Question?", step=1, max_steps=1))


def test_controller_rejects_non_decision_policy_output() -> None:
    with pytest.raises(TypeError, match="unsupported decision"):
        EvoRSController(SpyPolicy("visual")).decide(  # type: ignore[arg-type]
            observation("Question?")
        )


def test_agent_path_can_repeat_without_changing_required_order() -> None:
    decision = EvoRSController(FixedWorkflowPolicy()).decide(
        observation("Question?", path=("visual", "visual"))
    )
    assert decision == DispatchDecision(
        "reasoning",
        "Reason over scoped evidence and propose an answer candidate",
    )
