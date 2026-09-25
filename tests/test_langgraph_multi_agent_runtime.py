from __future__ import annotations

from rs_hmcomm.contracts import BudgetLimits, StopReason, TaskContext
from rs_hmcomm.control import (
    AllAgentPolicy,
    EvoRSController,
    FixedWorkflowPolicy,
    RuleRoutingPolicy,
)
from rs_hmcomm.experts import (
    KnowledgeExpert,
    ReasoningExpert,
    VerificationExpert,
    VisualExpert,
)
from rs_hmcomm.runtime import LangGraphMultiAgentRuntime


def experts():
    return {
        "visual": VisualExpert(),
        "reasoning": ReasoningExpert(),
        "verification": VerificationExpert(),
        "knowledge": KnowledgeExpert(),
    }


def task(question: str, answer: str = "6") -> TaskContext:
    return TaskContext(
        task_id="phase4-task",
        image_ref="fixture://phase4-image",
        question=question,
        task_type="scripted_phase4",
        answer_format="short_text",
        metadata={
            "scripted_fixture": True,
            "mock_answer": answer,
            "scientific_claim": False,
        },
    )


def budget(max_agent_calls: int = 8) -> BudgetLimits:
    return BudgetLimits(
        max_agent_calls=max_agent_calls,
        max_message_text_tokens=100000,
        max_message_structure_bytes=100000,
        max_message_latent_bytes=100000,
        max_errors=1,
    )


def runtime(policy, *, max_agent_calls: int = 8) -> LangGraphMultiAgentRuntime:
    return LangGraphMultiAgentRuntime(
        EvoRSController(policy),
        experts(),
        budget(max_agent_calls),
        max_steps=8,
    )


def test_rule_counting_routes_without_knowledge_and_finishes() -> None:
    result = runtime(RuleRoutingPolicy()).run(task("统计图中有多少架飞机？"))

    assert result.runtime == "langgraph"
    assert result.stop_reason == StopReason.COMPLETED
    assert result.answer == "6"
    assert result.agent_path == ("visual", "reasoning", "verification")
    assert "knowledge" not in result.agent_path
    assert result.total_cost.agent_calls == 3
    assert result.total_cost.message_structure_bytes > 0
    assert [item.receiver for item in result.messages] == [
        "reasoning",
        "verification",
    ]
    replay = result.replay()
    assert replay.agent_path == result.agent_path
    assert replay.answer_candidate_ref == result.answer_candidate_ref


def test_rule_airport_type_activates_visual_knowledge_reasoning() -> None:
    result = runtime(RuleRoutingPolicy()).run(
        task("判断这个机场类型", "civil_airport")
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert result.answer == "civil_airport"
    assert result.agent_path == (
        "visual",
        "knowledge",
        "reasoning",
        "verification",
    )
    assert any(item.kind == "domain_knowledge" for item in result.workspace.evidence)


def test_fixed_all_and_rule_share_one_runtime_and_budget_contract() -> None:
    shared_task = task("How many aircraft are visible?")
    fixed = runtime(FixedWorkflowPolicy()).run(shared_task, episode_id="fixed")
    all_agents = runtime(AllAgentPolicy()).run(shared_task, episode_id="all")
    rule = runtime(RuleRoutingPolicy()).run(shared_task, episode_id="rule")

    assert fixed.stop_reason == all_agents.stop_reason == rule.stop_reason
    assert fixed.agent_path == rule.agent_path == (
        "visual",
        "reasoning",
        "verification",
    )
    assert all_agents.agent_path == (
        "visual",
        "knowledge",
        "reasoning",
        "verification",
    )
    assert all(item.policy.endswith("Policy") for item in (fixed, all_agents, rule))


def test_budget_exhaustion_changes_the_executed_trajectory() -> None:
    result = runtime(RuleRoutingPolicy(), max_agent_calls=2).run(
        task("Count the aircraft")
    )

    assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert result.agent_path == ("visual", "reasoning")
    assert result.answer is None
    assert result.answer_candidate_ref is None
    assert result.total_cost.agent_calls == 2
    result.replay()
