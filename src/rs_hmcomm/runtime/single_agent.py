"""Budgeted Phase 1 runtime for one Generalist Agent."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping

from rs_hmcomm.agents.action_protocol import CallTool, Finish
from rs_hmcomm.agents.generalist import (
    GeneralistAgent,
    GeneralistDecision,
    GeneralistDecisionError,
    GeneralistTurnObservation,
)
from rs_hmcomm.backends.base import AgentOutput, BackendUsage, TokenAccounting, VisualInput
from rs_hmcomm.contracts import (
    AgentObservation,
    AgentResult,
    AgentResultStatus,
    AnswerCandidate,
    Assignment,
    BudgetLimits,
    CostVector,
    EpisodeEvent,
    EpisodeTrace,
    EventType,
    Evidence,
    StopReason,
    TaskContext,
)
from rs_hmcomm.evolution import (
    Candidate,
    EvolutionConfig,
    FitnessEvaluation,
    FitnessRecord,
    GenerationRecord,
    IntraAgentEvolution,
    VariationContext,
    VariationOutcome,
)
from rs_hmcomm.tools import ToolContext, ToolError, ToolRegistry, UnknownToolError


class _EvolutionBudgetStop(RuntimeError):
    pass


class _EvolutionInvalidProposal(RuntimeError):
    pass


class _EvolutionBackendFailure(RuntimeError):
    pass


def _finish_payload(action: Finish) -> dict[str, Any]:
    return {
        "answer": action.answer,
        "evidence_ids": list(action.evidence_ids),
        "confidence": action.confidence,
    }


def _finish_from_candidate(candidate: Candidate) -> Finish:
    payload = candidate.payload
    return Finish(
        answer=payload["answer"],
        evidence_ids=tuple(payload["evidence_ids"]),
        confidence=payload.get("confidence"),
    )


class _GeneralistAnswerVariation:
    """One runtime-mediated VLM revision used by the generic engine."""

    name = "generalist_vlm_answer_revision"

    def __init__(
        self,
        propose: Callable[[Candidate, VariationContext], tuple[Finish, CostVector]],
    ) -> None:
        self._propose = propose

    def vary(
        self,
        parent: Candidate,
        *,
        context: VariationContext,
    ) -> VariationOutcome:
        action, cost = self._propose(parent, context)
        return VariationOutcome(payload=_finish_payload(action), cost=cost)


class _EvidenceConfidenceFitness:
    """Label-free fitness for the first answer-evolution integration."""

    name = "evidence_confidence_fitness_v1"

    def __init__(self, visible_evidence_ids: frozenset[str]) -> None:
        self._visible_evidence_ids = visible_evidence_ids

    def evaluate(self, candidate: Candidate) -> FitnessEvaluation:
        action = _finish_from_candidate(candidate)
        cited = frozenset(action.evidence_ids)
        evidence_support = float(
            bool(cited) and cited.issubset(self._visible_evidence_ids)
        )
        confidence = action.confidence if action.confidence is not None else 0.0
        answer_format = float(bool(action.answer.strip()))
        score = 0.5 * evidence_support + 0.4 * confidence + 0.1 * answer_format
        return FitnessEvaluation(
            score=score,
            components={
                "evidence_support": evidence_support,
                "confidence": confidence,
                "answer_format": answer_format,
            },
        )


@dataclass(frozen=True)
class SingleAgentRunResult:
    task: TaskContext
    agent_result: AgentResult
    trace: EpisodeTrace
    answer: str | None
    stop_reason: StopReason
    model_id: str
    model_revision: str

    @property
    def cost(self) -> CostVector:
        return self.trace.total_cost

    @property
    def wall_time_ms(self) -> float | None:
        return self.trace.wall_time_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "agent_result": self.agent_result.to_dict(),
            "trace": self.trace.to_dict(),
            "answer": self.answer,
            "stop_reason": self.stop_reason.value,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "cost": self.cost.to_dict(),
            "wall_time_ms": self.wall_time_ms,
        }

    def validate_phase1_completion(self, *, require_tool: bool) -> None:
        """Validate the Phase 1 image→model→tool→evidence→answer chain."""

        state = self.trace.validate_complete()
        if state.stop_reason != StopReason.COMPLETED:
            raise ValueError("Phase 1 completion gate requires a completed trace")
        events = self.trace.events
        dispatches = [
            (index, event)
            for index, event in enumerate(events)
            if event.event_type == EventType.AGENT_DISPATCHED
        ]
        model_calls = [
            (index, event)
            for index, event in enumerate(events)
            if event.event_type == EventType.MODEL_CALLED
        ]
        if len(dispatches) != 1:
            raise ValueError("Phase 1 requires exactly one Agent dispatch")
        if not model_calls:
            raise ValueError("Phase 1 requires at least one model call")
        if dispatches[0][0] >= model_calls[0][0]:
            raise ValueError("Agent dispatch must precede model calls")

        tool_requests: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
        finish_actions = 0
        for index, event in enumerate(events):
            if event.event_type != EventType.AGENT_ACTED:
                continue
            action_type = event.action.get("action_type")
            if action_type == "call_tool":
                call_id = event.action.get("call_id")
                if isinstance(call_id, str):
                    tool_requests.setdefault(call_id, []).append((index, event.action))
            elif action_type == "finish":
                finish_actions += 1
        if finish_actions != 1:
            raise ValueError("Phase 1 completed trace requires exactly one finish action")

        successful_tool_calls: dict[str, int] = {}
        for index, event in enumerate(events):
            if event.event_type != EventType.TOOL_CALLED:
                continue
            call_id = event.action["call_id"]
            matching = [
                request_index
                for request_index, request in tool_requests.get(call_id, [])
                if request_index < index
                and request.get("tool_name") == event.action.get("tool_name")
            ]
            if not matching:
                raise ValueError("tool call has no preceding Agent request")
            tool_result = event.result.get("tool_result")
            if isinstance(tool_result, Mapping) and tool_result.get("ok") is True:
                successful_tool_calls[call_id] = index

        produced_evidence: set[str] = set()
        for index, event in enumerate(events):
            if event.event_type != EventType.RECORD_ADDED:
                continue
            source_call_id = event.action.get("source_call_id")
            records = event.result["records"]
            evidence_records = [
                record["record_id"]
                for record in records
                if record["record_type"] == "evidence"
            ]
            if evidence_records:
                if (
                    not isinstance(source_call_id, str)
                    or source_call_id not in successful_tool_calls
                    or successful_tool_calls[source_call_id] >= index
                ):
                    raise ValueError(
                        "evidence record does not reference a preceding successful tool call"
                    )
                produced_evidence.update(evidence_records)

        if require_tool:
            if not successful_tool_calls:
                raise ValueError("Phase 1 requires a successful tool call")
            if not produced_evidence:
                raise ValueError("Phase 1 requires evidence produced by that tool")
            if len(model_calls) < 2:
                raise ValueError("tool-using Phase 1 requires model calls before and after tool use")

        candidates = self.agent_result.answer_candidates
        if len(candidates) != 1:
            raise ValueError("Phase 1 requires exactly one answer candidate")
        candidate_evidence = set(candidates[0].evidence_ids)
        if not candidate_evidence or not candidate_evidence.issubset(produced_evidence):
            raise ValueError("answer candidate evidence is not produced by the tool chain")


class SingleAgentRuntime:
    """Execute a bounded tool-using loop without Workspace or multi-Agent state."""

    def __init__(
        self,
        agent: GeneralistAgent,
        tools: ToolRegistry,
        budget: BudgetLimits,
        *,
        max_steps: int = 4,
        evolution_config: EvolutionConfig | None = None,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.agent = agent
        self.tools = tools
        self.budget = budget
        self.max_steps = max_steps
        if evolution_config is not None and not isinstance(
            evolution_config,
            EvolutionConfig,
        ):
            raise TypeError("evolution_config must be an EvolutionConfig or None")
        if (
            evolution_config is not None
            and evolution_config.enabled
            and evolution_config.population_size != 2
        ):
            raise ValueError("Generalist answer evolution currently requires (1+1)")
        self.evolution_config = evolution_config

    def run(
        self,
        image: Any,
        question: str,
        *,
        sample_id: str = "sample-0",
        image_ref: str | None = None,
        task_type: str = "vqa",
        answer_format: str = "short_text",
        require_tool: bool = False,
        task_metadata: Mapping[str, Any] | None = None,
    ) -> SingleAgentRunResult:
        start = time.perf_counter()
        episode_id = f"phase1-{sample_id}"
        task = TaskContext(
            task_id=sample_id,
            image_ref=image_ref or f"runtime://{sample_id}",
            question=question,
            task_type=task_type,
            answer_format=answer_format,
            metadata=dict(task_metadata or {}),
        )
        assignment = Assignment(
            assignment_id="assignment-generalist-0",
            agent_id=self.agent.agent_id,
            objective="Produce an evidence-backed answer",
            created_step=0,
        )
        boundary_observation = AgentObservation(
            observation_id="observation-generalist-0",
            agent_id=self.agent.agent_id,
            task=task,
            assignment=assignment,
            step=0,
            budget_limits=self.budget,
        )
        visuals_by_id: dict[str, VisualInput] = {
            "input_image": VisualInput(
                visual_id="input_image",
                label="input_image",
                image=image,
            )
        }
        evidence: list[Evidence] = []
        recoverable_errors: list[str] = []
        events: list[EpisodeEvent] = []
        used_physical_call_ids: set[str] = set()
        model_id = type(self.agent.backend).__name__
        model_revision = ""
        evolution_summary: dict[str, Any] | None = None

        def elapsed_ms() -> float:
            return max(0.0, (time.perf_counter() - start) * 1000.0)

        def emit(
            event_type: EventType,
            actor: str,
            *,
            action: Mapping[str, Any] | None = None,
            result: Mapping[str, Any] | None = None,
            cost: CostVector | None = None,
        ) -> EpisodeEvent:
            event = EpisodeEvent(
                episode_id=episode_id,
                event_id=f"{episode_id}-event-{len(events)}",
                seq=len(events),
                event_type=event_type,
                actor=actor,
                action=dict(action or {}),
                result=dict(result or {}),
                cost_delta=cost or CostVector(),
                elapsed_ms=elapsed_ms(),
            )
            if event_type in {
                EventType.AGENT_DISPATCHED,
                EventType.MODEL_CALLED,
                EventType.TOOL_CALLED,
            }:
                call_id = event.action["call_id"]
                if call_id in used_physical_call_ids:
                    raise ValueError("physical call_id was reused inside the runtime")
                used_physical_call_ids.add(call_id)
            events.append(event)
            return event

        def trace() -> EpisodeTrace:
            return EpisodeTrace(
                episode_id,
                tuple(events),
                budget_limits=self.budget,
            )

        def budget_allows(delta: CostVector | None = None) -> bool:
            return self.budget.allows(
                trace().total_cost,
                delta,
                wall_time_ms=elapsed_ms(),
            )

        def terminate(
            reason: StopReason,
            *,
            failed: bool = False,
            answer_candidate: AnswerCandidate | None = None,
        ) -> SingleAgentRunResult:
            nonlocal model_id, model_revision
            if reason == StopReason.COMPLETED:
                status = AgentResultStatus.COMPLETED
                stop_reason = None
                answer_candidates = (answer_candidate,) if answer_candidate is not None else ()
            else:
                status = AgentResultStatus.FAILED if failed else AgentResultStatus.STOPPED
                stop_reason = reason
                answer_candidates = ()
            result_metadata: dict[str, Any] = {
                "model_id": model_id,
                "model_revision": model_revision,
                "internal_turns": sum(
                    event.event_type == EventType.MODEL_CALLED for event in events
                ),
            }
            if evolution_summary is not None:
                result_metadata["answer_evolution"] = evolution_summary
            agent_result = AgentResult(
                result_id="result-generalist-0",
                agent_id=self.agent.agent_id,
                assignment_id=assignment.assignment_id,
                status=status,
                evidence=tuple(evidence),
                answer_candidates=answer_candidates,
                stop_reason=stop_reason,
                metadata=result_metadata,
            )
            if reason == StopReason.COMPLETED:
                agent_result.validate_against(
                    boundary_observation,
                    observed_cost_delta=trace().total_cost,
                )
            else:
                agent_result.validate_against(boundary_observation)
            finish_action: dict[str, Any] = {}
            if answer_candidate is not None:
                finish_action["answer_candidate_ref"] = answer_candidate.candidate_id
            emit(
                EventType.EPISODE_FINISHED,
                "runtime",
                action=finish_action,
                result={"stop_reason": reason.value},
            )
            final_trace = trace()
            final_trace.validate_complete()
            return SingleAgentRunResult(
                task=task,
                agent_result=agent_result,
                trace=final_trace,
                answer=answer_candidate.answer if answer_candidate is not None else None,
                stop_reason=reason,
                model_id=model_id,
                model_revision=model_revision,
            )

        def model_cost(output: AgentOutput, fallback_latency_ms: float) -> CostVector:
            usage = output.usage
            billed = usage.latency_ms if usage.latency_ms > 0 else fallback_latency_ms
            return CostVector(
                model_calls=1,
                model_input_tokens=usage.input_tokens or 0,
                model_output_tokens=usage.output_tokens or 0,
                billed_compute_ms=billed,
            )

        def log_model_output(output: AgentOutput, fallback_latency_ms: float) -> None:
            nonlocal model_id, model_revision
            model_id = output.model_id or model_id
            model_revision = output.model_revision or model_revision
            output_hash = hashlib.sha256(output.text.encode("utf-8")).hexdigest()
            emit(
                EventType.MODEL_CALLED,
                model_id,
                action={"model_id": model_id},
                result={
                    "token_accounting": output.usage.token_accounting.value,
                    "usage": output.usage.to_dict(),
                    "output_sha256": output_hash,
                    "response_model_reported": output.structured.get(
                        "response_model_reported"
                    ),
                    "response_model_id": output.structured.get(
                        "response_model_id"
                    ),
                },
                cost=model_cost(output, fallback_latency_ms),
            )

        def log_evolution_event(event_name: str, record: object) -> None:
            if event_name == "candidate_proposed" and isinstance(record, Candidate):
                serialized_payload = json.dumps(
                    record.to_private_dict()["payload"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                emit(
                    EventType.CANDIDATE_PROPOSED,
                    self.agent.agent_id,
                    action={
                        "candidate_id": record.candidate_id,
                        "generation": record.generation,
                    },
                    result={
                        "parent_candidate_id": record.parent_id,
                        "variation_operator": record.variation_operator,
                        "payload_sha256": hashlib.sha256(
                            serialized_payload.encode("utf-8")
                        ).hexdigest(),
                        "scope": "private_agent_scratchpad",
                    },
                )
                return
            if event_name == "candidate_evaluated" and isinstance(
                record,
                FitnessRecord,
            ):
                emit(
                    EventType.CANDIDATE_EVALUATED,
                    self.agent.agent_id,
                    action={
                        "candidate_id": record.candidate_id,
                        "evaluator": record.evaluator,
                    },
                    result={
                        "score": record.evaluation.score,
                        "components": dict(record.evaluation.components),
                    },
                )
                return
            if event_name == "candidate_selected" and isinstance(
                record,
                GenerationRecord,
            ):
                emit(
                    EventType.CANDIDATE_SELECTED,
                    self.agent.agent_id,
                    action={
                        "generation": record.generation,
                        "strategy": record.selection.strategy,
                    },
                    result={
                        "selected_candidate_id": record.selection.selected_candidate_id,
                        "candidate_scores": dict(record.selection.candidate_scores),
                        "fitness_gain": record.fitness_gain,
                    },
                )
                return
            raise RuntimeError(f"unsupported evolution event: {event_name}")

        def record_failure(
            message: str,
            reason: StopReason,
            *,
            invalid_action: bool,
            can_retry: bool,
        ) -> StopReason | None:
            # Only callers may provide stable protocol text. Never pass an
            # arbitrary exception string here: event logs are public research
            # artifacts, not a private debug sink.
            safe_message = message.strip()[:500] or reason.value
            emit(
                EventType.ERROR,
                "runtime",
                action={"reason": reason.value},
                result={"message": safe_message},
                cost=CostVector(errors=1, invalid_actions=1 if invalid_action else 0),
            )
            if not budget_allows():
                return StopReason.BUDGET_EXHAUSTED
            retry_delta = CostVector(retries=1)
            if can_retry and budget_allows(retry_delta):
                emit(
                    EventType.WARNING,
                    "runtime",
                    action={"reason": "retry"},
                    result={"previous_error": safe_message},
                    cost=retry_delta,
                )
                recoverable_errors.append(safe_message)
                return None
            return reason

        emit(
            EventType.EPISODE_STARTED,
            "runtime",
            action={"task_id": task.task_id},
            result={
                "observation_id": boundary_observation.observation_id,
                "budget_limits": self.budget.to_dict(),
            },
        )
        dispatch_cost = CostVector(agent_calls=1)
        if not budget_allows(dispatch_cost):
            return terminate(StopReason.BUDGET_EXHAUSTED)
        emit(
            EventType.AGENT_DISPATCHED,
            "runtime",
            action={"agent_id": self.agent.agent_id, "assignment_id": assignment.assignment_id},
            result={"policy": "single_agent"},
            cost=dispatch_cost,
        )

        for internal_step in range(self.max_steps):
            if not budget_allows(CostVector(model_calls=1)):
                return terminate(StopReason.BUDGET_EXHAUSTED)
            turn = GeneralistTurnObservation(
                task=task,
                step=internal_step,
                visuals=tuple(visuals_by_id.values()),
                evidence=tuple(evidence),
                recoverable_errors=tuple(recoverable_errors),
                require_tool=require_tool,
            )
            model_started = time.perf_counter()
            try:
                decision = self.agent.decide(turn)
            except GeneralistDecisionError as exc:
                call_elapsed = (time.perf_counter() - model_started) * 1000.0
                log_model_output(exc.backend_output, call_elapsed)
                reason = record_failure(
                    "model action failed protocol validation",
                    StopReason.INVALID_ACTION,
                    invalid_action=True,
                    can_retry=internal_step + 1 < self.max_steps,
                )
                if reason is not None:
                    return terminate(reason, failed=reason != StopReason.BUDGET_EXHAUSTED)
                continue
            except Exception as exc:
                call_elapsed = max(0.0, (time.perf_counter() - model_started) * 1000.0)
                fallback_usage = BackendUsage(
                    latency_ms=call_elapsed,
                    stop_reason="backend_error",
                    token_accounting=TokenAccounting.UNAVAILABLE,
                )
                fallback_output = AgentOutput(
                    text="",
                    usage=fallback_usage,
                    model_id=model_id,
                    model_revision=model_revision,
                )
                log_model_output(fallback_output, call_elapsed)
                reason = record_failure(
                    f"backend error ({type(exc).__name__})",
                    StopReason.BACKEND_ERROR,
                    invalid_action=False,
                    can_retry=False,
                )
                return terminate(reason or StopReason.BACKEND_ERROR, failed=True)

            call_elapsed = max(0.0, (time.perf_counter() - model_started) * 1000.0)
            log_model_output(decision.backend_output, call_elapsed)
            if not budget_allows():
                return terminate(StopReason.BUDGET_EXHAUSTED)

            action = decision.action
            if isinstance(action, CallTool):
                emit(
                    EventType.AGENT_ACTED,
                    self.agent.agent_id,
                    action={
                        "action_type": "call_tool",
                        "call_id": action.call_id,
                        "tool_name": action.tool_name,
                    },
                    result={"status": "requested"},
                )
                if action.call_id in used_physical_call_ids:
                    reason = record_failure(
                        "tool call_id is not unique within this episode",
                        StopReason.INVALID_ACTION,
                        invalid_action=True,
                        can_retry=internal_step + 1 < self.max_steps,
                    )
                    if reason is not None:
                        return terminate(
                            reason,
                            failed=reason != StopReason.BUDGET_EXHAUSTED,
                        )
                    continue
                try:
                    tool = self.tools.get(action.tool_name)
                except UnknownToolError:
                    reason = record_failure(
                        "requested tool is not registered",
                        StopReason.INVALID_ACTION,
                        invalid_action=True,
                        can_retry=internal_step + 1 < self.max_steps,
                    )
                    if reason is not None:
                        return terminate(reason, failed=reason != StopReason.BUDGET_EXHAUSTED)
                    continue
                if not budget_allows(CostVector(tool_calls=1)):
                    return terminate(StopReason.BUDGET_EXHAUSTED)

                tool_started = time.perf_counter()
                try:
                    tool_result = tool.execute(
                        action,
                        ToolContext(
                            agent_id=self.agent.agent_id,
                            step=0,
                            visuals_by_id=visuals_by_id,
                        ),
                    )
                except ToolError as exc:
                    tool_elapsed = max(0.0, (time.perf_counter() - tool_started) * 1000.0)
                    emit(
                        EventType.TOOL_CALLED,
                        action.tool_name,
                        action={"tool_name": action.tool_name, "call_id": action.call_id},
                        result={"ok": False, "error_type": type(exc).__name__},
                        cost=CostVector(tool_calls=1, billed_compute_ms=tool_elapsed),
                    )
                    reason = record_failure(
                        f"tool rejected the request ({type(exc).__name__})",
                        StopReason.TOOL_ERROR,
                        invalid_action=False,
                        can_retry=internal_step + 1 < self.max_steps,
                    )
                    if reason is not None:
                        return terminate(reason, failed=reason != StopReason.BUDGET_EXHAUSTED)
                    continue
                except Exception as exc:
                    tool_elapsed = max(0.0, (time.perf_counter() - tool_started) * 1000.0)
                    emit(
                        EventType.TOOL_CALLED,
                        action.tool_name,
                        action={"tool_name": action.tool_name, "call_id": action.call_id},
                        result={"ok": False, "error_type": type(exc).__name__},
                        cost=CostVector(tool_calls=1, billed_compute_ms=tool_elapsed),
                    )
                    reason = record_failure(
                        f"tool error ({type(exc).__name__})",
                        StopReason.TOOL_ERROR,
                        invalid_action=False,
                        can_retry=False,
                    )
                    return terminate(reason or StopReason.TOOL_ERROR, failed=True)

                tool_elapsed = max(0.0, (time.perf_counter() - tool_started) * 1000.0)
                emit(
                    EventType.TOOL_CALLED,
                    action.tool_name,
                    action={"tool_name": action.tool_name, "call_id": action.call_id},
                    result={"tool_result": tool_result.to_log_dict()},
                    cost=CostVector(tool_calls=1, billed_compute_ms=tool_elapsed),
                )
                for visual in tool_result.visuals:
                    if visual.visual_id in visuals_by_id:
                        return terminate(StopReason.TOOL_ERROR, failed=True)
                    visuals_by_id[visual.visual_id] = visual
                existing_evidence_ids = {item.evidence_id for item in evidence}
                if any(item.evidence_id in existing_evidence_ids for item in tool_result.evidence):
                    return terminate(StopReason.TOOL_ERROR, failed=True)
                evidence.extend(tool_result.evidence)
                if tool_result.evidence:
                    emit(
                        EventType.RECORD_ADDED,
                        "runtime",
                        action={"source_call_id": action.call_id},
                        result={
                            "records": [
                                {
                                    "record_id": item.evidence_id,
                                    "record_type": "evidence",
                                }
                                for item in tool_result.evidence
                            ]
                        },
                    )
                if not budget_allows():
                    return terminate(StopReason.BUDGET_EXHAUSTED)
                continue

            if isinstance(action, Finish):
                evidence_ids = {item.evidence_id for item in evidence}
                missing = set(action.evidence_ids) - evidence_ids
                needs_tool = require_tool and not any(item.kind == "crop" for item in evidence)
                if missing or needs_tool:
                    message = (
                        "finish references evidence that is not visible"
                        if missing
                        else "finish is not allowed before the required crop tool"
                    )
                    reason = record_failure(
                        message,
                        StopReason.INVALID_ACTION,
                        invalid_action=True,
                        can_retry=internal_step + 1 < self.max_steps,
                    )
                    if reason is not None:
                        return terminate(reason, failed=reason != StopReason.BUDGET_EXHAUSTED)
                    continue
                selected_action = action
                record_source = "finish"
                if (
                    self.evolution_config is not None
                    and self.evolution_config.enabled
                ):
                    seed_candidate = Candidate(
                        candidate_id="private-answer-generalist-seed",
                        payload=_finish_payload(action),
                    )

                    def propose_variation(
                        parent: Candidate,
                        context: VariationContext,
                    ) -> tuple[Finish, CostVector]:
                        if not budget_allows(CostVector(model_calls=1)):
                            raise _EvolutionBudgetStop
                        parent_action = _finish_from_candidate(parent)
                        model_started = time.perf_counter()
                        try:
                            decision = self.agent.propose_answer_variation(
                                turn,
                                parent_answer=parent_action.answer,
                                parent_evidence_ids=parent_action.evidence_ids,
                                parent_confidence=parent_action.confidence,
                            )
                        except GeneralistDecisionError as exc:
                            call_elapsed = max(
                                0.0,
                                (time.perf_counter() - model_started) * 1000.0,
                            )
                            log_model_output(exc.backend_output, call_elapsed)
                            if not budget_allows():
                                raise _EvolutionBudgetStop from exc
                            raise
                        except Exception as exc:
                            call_elapsed = max(
                                0.0,
                                (time.perf_counter() - model_started) * 1000.0,
                            )
                            fallback_output = AgentOutput(
                                text="",
                                usage=BackendUsage(
                                    latency_ms=call_elapsed,
                                    stop_reason="backend_error",
                                    token_accounting=TokenAccounting.UNAVAILABLE,
                                ),
                                model_id=model_id,
                                model_revision=model_revision,
                            )
                            log_model_output(fallback_output, call_elapsed)
                            if not budget_allows():
                                raise _EvolutionBudgetStop from exc
                            raise _EvolutionBackendFailure from exc

                        call_elapsed = max(
                            0.0,
                            (time.perf_counter() - model_started) * 1000.0,
                        )
                        log_model_output(decision.backend_output, call_elapsed)
                        if not budget_allows():
                            raise _EvolutionBudgetStop
                        proposal = decision.action
                        missing_proposal_evidence = (
                            set(proposal.evidence_ids) - evidence_ids
                        )
                        if missing_proposal_evidence:
                            raise _EvolutionInvalidProposal
                        return (
                            proposal,
                            model_cost(decision.backend_output, call_elapsed),
                        )

                    evolution = IntraAgentEvolution(
                        _GeneralistAnswerVariation(propose_variation),
                        _EvidenceConfidenceFitness(frozenset(evidence_ids)),
                        self.evolution_config,
                    )
                    try:
                        evolution_result = evolution.run(
                            seed_candidate,
                            event_hook=log_evolution_event,
                        )
                    except _EvolutionBudgetStop:
                        return terminate(StopReason.BUDGET_EXHAUSTED)
                    except GeneralistDecisionError:
                        reason = record_failure(
                            "answer evolution failed protocol validation",
                            StopReason.INVALID_ACTION,
                            invalid_action=True,
                            can_retry=False,
                        )
                        return terminate(
                            reason or StopReason.INVALID_ACTION,
                            failed=reason != StopReason.BUDGET_EXHAUSTED,
                        )
                    except _EvolutionInvalidProposal:
                        reason = record_failure(
                            "answer evolution cited evidence that is not visible",
                            StopReason.INVALID_ACTION,
                            invalid_action=True,
                            can_retry=False,
                        )
                        return terminate(
                            reason or StopReason.INVALID_ACTION,
                            failed=reason != StopReason.BUDGET_EXHAUSTED,
                        )
                    except _EvolutionBackendFailure:
                        reason = record_failure(
                            "answer evolution backend error",
                            StopReason.BACKEND_ERROR,
                            invalid_action=False,
                            can_retry=False,
                        )
                        return terminate(
                            reason or StopReason.BACKEND_ERROR,
                            failed=True,
                        )

                    selected_action = _finish_from_candidate(
                        evolution_result.selected_candidate
                    )
                    evolution_summary = evolution_result.to_dict()
                    record_source = "evolution_selection"
                emit(
                    EventType.AGENT_ACTED,
                    self.agent.agent_id,
                    action={"action_type": "finish"},
                    result={
                        "status": "requested",
                        "source": record_source,
                        "answer_sha256": hashlib.sha256(
                            selected_action.answer.encode("utf-8")
                        ).hexdigest(),
                    },
                )
                answer_candidate = AnswerCandidate(
                    candidate_id="answer-generalist-0",
                    answer=selected_action.answer,
                    created_by=self.agent.agent_id,
                    evidence_ids=selected_action.evidence_ids,
                    created_step=0,
                    confidence=selected_action.confidence,
                )
                if not budget_allows():
                    return terminate(StopReason.BUDGET_EXHAUSTED)
                emit(
                    EventType.RECORD_ADDED,
                    "runtime",
                    action={"source": record_source},
                    result={
                        "records": [
                            {
                                "record_id": answer_candidate.candidate_id,
                                "record_type": "answer_candidate",
                                "answer_sha256": hashlib.sha256(
                                    answer_candidate.answer.encode("utf-8")
                                ).hexdigest(),
                                "evidence_ids": list(answer_candidate.evidence_ids),
                            }
                        ]
                    },
                )
                return terminate(
                    StopReason.COMPLETED,
                    answer_candidate=answer_candidate,
                )

            reason = record_failure(
                f"unsupported action object: {type(action).__name__}",
                StopReason.INVALID_ACTION,
                invalid_action=True,
                can_retry=False,
            )
            return terminate(reason or StopReason.INVALID_ACTION, failed=True)

        return terminate(StopReason.MAX_STEPS)


__all__ = ["SingleAgentRunResult", "SingleAgentRuntime"]
