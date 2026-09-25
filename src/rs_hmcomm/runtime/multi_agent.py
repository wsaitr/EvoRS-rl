"""LangGraph runtime for controlled, non-RL Cognitive Expert collaboration."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, TypedDict

from langgraph.graph import END, START, StateGraph

from rs_hmcomm.communication import (
    ContextRouter,
    Inbox,
    MessageEnvelope,
    MessageModality,
    ObservationBuilder,
)
from rs_hmcomm.contracts import BudgetLimits, CostVector, StopReason, TaskContext
from rs_hmcomm.control import (
    AbortDecision,
    ControllerDecision,
    ControllerObservation,
    ControllerWorkspaceSummary,
    DispatchDecision,
    EvoRSController,
    FinishDecision,
)
from rs_hmcomm.experts import CognitiveExpert, ExpertObservation, EXPERT_IDS
from rs_hmcomm.workspace import WorkspaceReducer, WorkspaceState


def _decision_dict(decision: ControllerDecision) -> dict[str, Any]:
    if isinstance(decision, DispatchDecision):
        payload = {
            "type": "dispatch",
            "expert_id": decision.expert_id,
            "objective": decision.objective,
            "context_spec": decision.context_spec.to_dict(),
            "local_budget": (
                decision.local_budget.to_dict()
                if decision.local_budget is not None
                else None
            ),
        }
        if decision.policy_action_id is not None:
            payload["policy_action_id"] = decision.policy_action_id
        return payload
    if isinstance(decision, FinishDecision):
        payload = {
            "type": "finish",
            "answer_candidate_ref": decision.answer_candidate_ref,
        }
        if decision.policy_action_id is not None:
            payload["policy_action_id"] = decision.policy_action_id
        return payload
    payload = {"type": "abort", "stop_reason": decision.stop_reason.value}
    if decision.policy_action_id is not None:
        payload["policy_action_id"] = decision.policy_action_id
    return payload


@dataclass(frozen=True, slots=True)
class MultiAgentTransition:
    step: int
    policy: str
    controller_observation_sha256: str
    decision: Mapping[str, Any]
    active_expert: str | None
    workspace_version_before: int
    workspace_version_after: int
    message_ids: tuple[str, ...]
    cost_delta: CostVector
    expert_metrics: Mapping[str, Any] = field(default_factory=dict)
    stop_reason: StopReason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "policy": self.policy,
            "controller_observation_sha256": self.controller_observation_sha256,
            "decision": dict(self.decision),
            "active_expert": self.active_expert,
            "workspace_version_before": self.workspace_version_before,
            "workspace_version_after": self.workspace_version_after,
            "message_ids": list(self.message_ids),
            "cost_delta": self.cost_delta.to_dict(),
            "expert_metrics": dict(self.expert_metrics),
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }


@dataclass(frozen=True, slots=True)
class MultiAgentReplay:
    agent_path: tuple[str, ...]
    disclosed_message_ids: tuple[str, ...]
    total_cost: CostVector
    stop_reason: StopReason
    answer_candidate_ref: str | None


@dataclass(frozen=True, slots=True)
class MultiAgentRunResult:
    task: TaskContext
    workspace: WorkspaceState
    policy: str
    agent_path: tuple[str, ...]
    messages: tuple[MessageEnvelope, ...]
    transitions: tuple[MultiAgentTransition, ...]
    total_cost: CostVector
    stop_reason: StopReason
    answer_candidate_ref: str | None
    answer: str | None
    capability_requests: tuple[Any, ...] = ()
    runtime: str = "langgraph"

    def replay(self) -> MultiAgentReplay:
        dispatches = tuple(
            transition.active_expert
            for transition in self.transitions
            if transition.active_expert is not None
        )
        if dispatches != self.agent_path:
            raise ValueError("transition dispatch path does not match agent_path")
        expected_version = 0
        reconstructed_cost = CostVector()
        message_ids: list[str] = []
        for transition in self.transitions:
            if transition.active_expert is not None:
                if transition.workspace_version_before != expected_version:
                    raise ValueError("workspace transition versions are not contiguous")
                if transition.workspace_version_after != expected_version + 1:
                    raise ValueError("an expert transition must advance one workspace version")
                expected_version += 1
            elif (
                transition.workspace_version_before != expected_version
                or transition.workspace_version_after != expected_version
            ):
                raise ValueError("terminal transitions must not mutate the workspace")
            reconstructed_cost = reconstructed_cost + transition.cost_delta
            message_ids.extend(transition.message_ids)
        if expected_version != self.workspace.version:
            raise ValueError("replayed workspace version does not match final workspace")
        if reconstructed_cost != self.total_cost:
            raise ValueError("replayed cost does not match total_cost")
        replayed_inbox = Inbox.replay(self.messages)
        persisted_message_ids = tuple(item.message_id for item in replayed_inbox.messages)
        if tuple(message_ids) != persisted_message_ids:
            raise ValueError("transition disclosures do not match persisted messages")
        if self.stop_reason == StopReason.COMPLETED:
            candidates = {
                item.candidate_id: item for item in self.workspace.answer_candidates
            }
            if self.answer_candidate_ref not in candidates:
                raise ValueError("completed run has no resolvable answer candidate")
            if candidates[self.answer_candidate_ref].answer != self.answer:
                raise ValueError("completed answer does not match selected candidate")
        elif self.answer_candidate_ref is not None or self.answer is not None:
            raise ValueError("non-completed run must not expose an answer")
        return MultiAgentReplay(
            agent_path=self.agent_path,
            disclosed_message_ids=persisted_message_ids,
            total_cost=self.total_cost,
            stop_reason=self.stop_reason,
            answer_candidate_ref=self.answer_candidate_ref,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "policy": self.policy,
            "task": self.task.to_dict(),
            "workspace": self.workspace.to_dict(),
            "agent_path": list(self.agent_path),
            "messages": [item.to_dict() for item in self.messages],
            "transitions": [item.to_dict() for item in self.transitions],
            "total_cost": self.total_cost.to_dict(),
            "stop_reason": self.stop_reason.value,
            "answer_candidate_ref": self.answer_candidate_ref,
            "answer": self.answer,
            "capability_requests": [item.to_dict() for item in self.capability_requests],
        }


class _GraphState(TypedDict):
    workspace: WorkspaceState
    inbox: Inbox
    decision: ControllerDecision | None
    decision_observation_sha256: str
    agent_path: tuple[str, ...]
    transitions: tuple[MultiAgentTransition, ...]
    total_cost: CostVector
    pending_message_ids: tuple[str, ...]
    pending_message_cost: CostVector
    answer_candidate_ref: str | None
    answer: str | None
    stop_reason: StopReason | None
    capability_requests: tuple[Any, ...]


class LangGraphMultiAgentRuntime:
    """Run one explicitly routed Expert at a time through LangGraph."""

    def __init__(
        self,
        controller: EvoRSController,
        experts: Mapping[str, CognitiveExpert],
        budget: BudgetLimits,
        *,
        max_steps: int = 8,
        disclosure_mode: str = "scoped",
    ) -> None:
        if disclosure_mode not in ("full", "scoped"):
            raise ValueError("disclosure_mode must be 'full' or 'scoped'")
        if not isinstance(controller, EvoRSController):
            raise TypeError("controller must be an EvoRSController")
        if not isinstance(budget, BudgetLimits):
            raise TypeError("budget must be BudgetLimits")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        normalized = dict(experts)
        if set(normalized) != set(EXPERT_IDS):
            raise ValueError(f"experts must contain exactly {sorted(EXPERT_IDS)}")
        for expert_id, expert in normalized.items():
            if getattr(expert, "expert_id", None) != expert_id:
                raise ValueError(f"expert registry key does not match {expert_id}")
            if not callable(getattr(expert, "run", None)):
                raise TypeError(f"expert {expert_id} must implement run")
        self.controller = controller
        self.experts = normalized
        self.budget = budget
        self.max_steps = max_steps
        self.disclosure_mode = disclosure_mode
        self.policy_name = type(controller.policy).__name__

    @staticmethod
    def _controller_digest(observation: ControllerObservation) -> str:
        payload = {
            "task_id": observation.task.task_id,
            "question": observation.task.question,
            "workspace_version": observation.workspace.version,
            "workspace_summary": observation.workspace.to_dict(),
            "agent_path": list(observation.agent_path),
            "step": observation.step,
            "max_steps": observation.max_steps,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _build_graph(self, router: ContextRouter):
        def controller_node(state: _GraphState) -> dict[str, Any]:
            if state["stop_reason"] is not None:
                return {"decision": AbortDecision(state["stop_reason"])}
            observation = ControllerObservation(
                task=state["workspace"].task,
                workspace=ControllerWorkspaceSummary.from_workspace(
                    state["workspace"],
                    message_count=len(state["inbox"]),
                    total_cost=state["total_cost"],
                ),
                agent_path=state["agent_path"],
                step=len(state["agent_path"]),
                max_steps=self.max_steps,
                capability_requests=state["capability_requests"],
            )
            digest = self._controller_digest(observation)
            decision = self.controller.decide(observation)
            if not isinstance(decision, DispatchDecision):
                return {
                    "decision": decision,
                    "decision_observation_sha256": digest,
                    "pending_message_ids": (),
                    "pending_message_cost": CostVector(),
                }

            all_refs = state["workspace"].record_ids - {state["workspace"].task.task_id}
            if self.disclosure_mode == "full":
                non_task_refs = tuple(sorted(all_refs))
                modalities = [MessageModality.TEXT, MessageModality.STRUCTURE]
                if state["workspace"].artifacts:
                    modalities.append(MessageModality.LATENT)
            else:
                spec = decision.context_spec
                requested_refs = tuple(
                    dict.fromkeys((*spec.payload_refs, *spec.focus_node_ids))
                )
                if len(requested_refs) > spec.max_items:
                    raise ValueError("context_spec exceeds max_items")
                non_task_refs = tuple(sorted(requested_refs))
                modalities = [MessageModality(item) for item in spec.modalities]
            next_inbox = Inbox.replay(state["inbox"].messages)
            message_ids: tuple[str, ...] = ()
            message_cost = CostVector()
            if non_task_refs:
                envelope, grant = router.route(
                    state["workspace"],
                    "evors-controller",
                    decision.expert_id,
                    non_task_refs,
                    modalities,
                    len(state["agent_path"]),
                    scope="controller_dispatch",
                )
                message_cost = grant.cost
                message_ids = (envelope.message_id,)
                if not self.budget.allows(
                    state["total_cost"],
                    message_cost + CostVector(agent_calls=1),
                ):
                    return {
                        "decision": AbortDecision(StopReason.BUDGET_EXHAUSTED),
                        "decision_observation_sha256": digest,
                        "pending_message_ids": (),
                        "pending_message_cost": CostVector(),
                    }
                next_inbox.append(envelope)
            elif not self.budget.allows(
                state["total_cost"],
                CostVector(agent_calls=1),
            ):
                return {
                    "decision": AbortDecision(StopReason.BUDGET_EXHAUSTED),
                    "decision_observation_sha256": digest,
                    "pending_message_ids": (),
                    "pending_message_cost": CostVector(),
                }
            return {
                "decision": decision,
                "decision_observation_sha256": digest,
                "inbox": next_inbox,
                "pending_message_ids": message_ids,
                "pending_message_cost": message_cost,
            }

        def route_decision(state: _GraphState) -> str:
            decision = state["decision"]
            if isinstance(decision, DispatchDecision):
                return decision.expert_id
            if isinstance(decision, FinishDecision):
                return "finish"
            return "abort"

        def expert_node(expert_id: str):
            def execute(state: _GraphState) -> dict[str, Any]:
                decision = state["decision"]
                if not isinstance(decision, DispatchDecision):
                    raise RuntimeError("expert node requires a DispatchDecision")
                before = state["workspace"]
                view = ObservationBuilder.build(before, expert_id, state["inbox"])
                observation = ExpertObservation(
                    episode_id=before.episode_id,
                    step=len(state["agent_path"]),
                    expert_id=expert_id,
                    objective=decision.objective,
                    workspace=view,
                    local_budget=decision.local_budget,
                )
                dispatch_cost = state["pending_message_cost"] + CostVector(agent_calls=1)
                try:
                    result = self.experts[expert_id].run(observation)
                    result.validate_against(observation)
                    measured_cost = dispatch_cost + result.cost_delta
                    if decision.local_budget is not None and not decision.local_budget.allows(
                        CostVector(), result.cost_delta
                    ):
                        transition = MultiAgentTransition(
                            step=len(state["agent_path"]),
                            policy=self.policy_name,
                            controller_observation_sha256=state["decision_observation_sha256"],
                            decision=_decision_dict(decision),
                            active_expert=None,
                            workspace_version_before=before.version,
                            workspace_version_after=before.version,
                            message_ids=state["pending_message_ids"],
                            cost_delta=measured_cost,
                            stop_reason=StopReason.BUDGET_EXHAUSTED,
                        )
                        return {
                            "transitions": state["transitions"] + (transition,),
                            "total_cost": state["total_cost"] + measured_cost,
                            "stop_reason": StopReason.BUDGET_EXHAUSTED,
                            "pending_message_ids": (),
                            "pending_message_cost": CostVector(),
                        }
                    if not self.budget.allows(state["total_cost"], measured_cost):
                        transition = MultiAgentTransition(
                            step=len(state["agent_path"]),
                            policy=self.policy_name,
                            controller_observation_sha256=state["decision_observation_sha256"],
                            decision=_decision_dict(decision),
                            active_expert=None,
                            workspace_version_before=before.version,
                            workspace_version_after=before.version,
                            message_ids=state["pending_message_ids"],
                            cost_delta=measured_cost,
                            stop_reason=StopReason.BUDGET_EXHAUSTED,
                        )
                        return {
                            "transitions": state["transitions"] + (transition,),
                            "total_cost": state["total_cost"] + measured_cost,
                            "stop_reason": StopReason.BUDGET_EXHAUSTED,
                            "pending_message_ids": (),
                            "pending_message_cost": CostVector(),
                        }
                    after = WorkspaceReducer.apply(before, result.workspace_delta)
                except Exception as exc:  # preserve an auditable atomic rollback
                    error_cost = dispatch_cost + CostVector(errors=1)
                    transition = MultiAgentTransition(
                        step=len(state["agent_path"]),
                        policy=self.policy_name,
                        controller_observation_sha256=state[
                            "decision_observation_sha256"
                        ],
                        decision=_decision_dict(decision),
                        active_expert=None,
                        workspace_version_before=before.version,
                        workspace_version_after=before.version,
                        message_ids=state["pending_message_ids"],
                        cost_delta=error_cost,
                        expert_metrics={
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "workspace_rollback": True,
                        },
                        stop_reason=StopReason.NO_PROGRESS,
                    )
                    return {
                        "transitions": state["transitions"] + (transition,),
                        "total_cost": state["total_cost"] + error_cost,
                        "stop_reason": StopReason.NO_PROGRESS,
                        "pending_message_ids": (),
                        "pending_message_cost": CostVector(),
                    }
                transition = MultiAgentTransition(
                    step=len(state["agent_path"]),
                    policy=self.policy_name,
                    controller_observation_sha256=state[
                        "decision_observation_sha256"
                    ],
                    decision=_decision_dict(decision),
                    active_expert=expert_id,
                    workspace_version_before=before.version,
                    workspace_version_after=after.version,
                    message_ids=state["pending_message_ids"],
                    cost_delta=measured_cost,
                    expert_metrics=result.metrics,
                )
                requests = state["capability_requests"]
                if result.capability_request is not None:
                    requests = requests + (result.capability_request,)
                return {
                    "workspace": after,
                    "agent_path": state["agent_path"] + (expert_id,),
                    "transitions": state["transitions"] + (transition,),
                    "total_cost": state["total_cost"] + measured_cost,
                    "capability_requests": requests,
                    "pending_message_ids": (),
                    "pending_message_cost": CostVector(),
                }

            return execute

        def finish_node(state: _GraphState) -> dict[str, Any]:
            decision = state["decision"]
            if not isinstance(decision, FinishDecision):
                raise RuntimeError("finish node requires a FinishDecision")
            candidates = {
                item.candidate_id: item for item in state["workspace"].answer_candidates
            }
            candidate = candidates[decision.answer_candidate_ref]
            transition = MultiAgentTransition(
                step=len(state["agent_path"]),
                policy=self.policy_name,
                controller_observation_sha256=state["decision_observation_sha256"],
                decision=_decision_dict(decision),
                active_expert=None,
                workspace_version_before=state["workspace"].version,
                workspace_version_after=state["workspace"].version,
                message_ids=(),
                cost_delta=CostVector(),
                stop_reason=StopReason.COMPLETED,
            )
            return {
                "transitions": state["transitions"] + (transition,),
                "answer_candidate_ref": candidate.candidate_id,
                "answer": candidate.answer,
                "stop_reason": StopReason.COMPLETED,
            }

        def abort_node(state: _GraphState) -> dict[str, Any]:
            decision = state["decision"]
            if not isinstance(decision, AbortDecision):
                raise RuntimeError("abort node requires an AbortDecision")
            transition = MultiAgentTransition(
                step=len(state["agent_path"]),
                policy=self.policy_name,
                controller_observation_sha256=state["decision_observation_sha256"],
                decision=_decision_dict(decision),
                active_expert=None,
                workspace_version_before=state["workspace"].version,
                workspace_version_after=state["workspace"].version,
                message_ids=(),
                cost_delta=CostVector(),
                stop_reason=decision.stop_reason,
            )
            return {
                "transitions": state["transitions"] + (transition,),
                "stop_reason": decision.stop_reason,
            }

        builder = StateGraph(_GraphState)
        builder.add_node("controller", controller_node)
        for expert_id in sorted(EXPERT_IDS):
            builder.add_node(expert_id, expert_node(expert_id))
            builder.add_edge(expert_id, "controller")
        builder.add_node("finish", finish_node)
        builder.add_node("abort", abort_node)
        builder.add_edge(START, "controller")
        builder.add_conditional_edges("controller", route_decision)
        builder.add_edge("finish", END)
        builder.add_edge("abort", END)
        return builder.compile()

    def run(self, task: TaskContext, *, episode_id: str | None = None) -> MultiAgentRunResult:
        if not isinstance(task, TaskContext):
            raise TypeError("task must be a TaskContext")
        initial_workspace = WorkspaceState.initial(
            episode_id or f"episode-{task.task_id}",
            task,
        )
        graph = self._build_graph(ContextRouter())
        initial: _GraphState = {
            "workspace": initial_workspace,
            "inbox": Inbox(),
            "decision": None,
            "decision_observation_sha256": "",
            "agent_path": (),
            "transitions": (),
            "total_cost": CostVector(),
            "pending_message_ids": (),
            "pending_message_cost": CostVector(),
            "answer_candidate_ref": None,
            "answer": None,
            "stop_reason": None,
            "capability_requests": (),
        }
        final = graph.invoke(
            initial,
            {"recursion_limit": self.max_steps * 3 + 10},
        )
        stop_reason = final["stop_reason"]
        if stop_reason is None:
            raise RuntimeError("LangGraph collaboration ended without a stop reason")
        result = MultiAgentRunResult(
            task=task,
            workspace=final["workspace"],
            policy=self.policy_name,
            agent_path=tuple(final["agent_path"]),
            messages=final["inbox"].messages,
            transitions=tuple(final["transitions"]),
            total_cost=final["total_cost"],
            stop_reason=stop_reason,
            answer_candidate_ref=final["answer_candidate_ref"],
            answer=final["answer"],
            capability_requests=tuple(final["capability_requests"]),
        )
        result.replay()
        return result


__all__ = [
    "LangGraphMultiAgentRuntime",
    "MultiAgentReplay",
    "MultiAgentRunResult",
    "MultiAgentTransition",
]
