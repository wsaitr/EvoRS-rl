"""Real Phase 5 controller-RL training and reward contracts.

This module trains only :class:`LearnedRoutingPolicy`.  The LangGraph runtime
remains the environment and Cognitive Experts remain frozen environment
components.  Every sampled action is audited against the transition it caused.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
import math
import random
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from rs_hmcomm.contracts import StopReason, TaskContext
from rs_hmcomm.control import LearnedRoutingPolicy, RoutingSample
from rs_hmcomm.runtime import MultiAgentRunResult


REWARD_SCHEMA = "evors.routing-reward.v1"
TRAINING_SCHEMA = "evors.routing-training.v1"


def _normalized_answer(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _integer_answer(value: str) -> int | None:
    match = re.search(r"[-+]?\d+", str(value).replace(",", ""))
    return int(match.group()) if match else None


def score_answer(task_type: str, prediction: str, reference: str) -> float:
    """Deterministic exact task reward used only by the environment."""

    lowered = task_type.casefold()
    if any(token in lowered for token in ("quantity", "count")):
        predicted = _integer_answer(prediction)
        expected = _integer_answer(reference)
        return float(predicted is not None and expected is not None and predicted == expected)
    if any(token in lowered for token in ("exist", "presence")):
        yes = {"yes", "true", "present", "有", "是", "存在"}
        no = {"no", "false", "absent", "无", "否", "不存在"}
        left = _normalized_answer(prediction).rstrip(".。")
        right = _normalized_answer(reference).rstrip(".。")
        if left in yes and right in yes:
            return 1.0
        if left in no and right in no:
            return 1.0
        return 0.0
    return float(_normalized_answer(prediction) == _normalized_answer(reference))


@dataclass(frozen=True, slots=True)
class RoutingRewardWeights:
    task: float = 1.0
    evidence_gain: float = 0.04
    structure_gain: float = 0.04
    message_text_token: float = 0.00005
    message_structure_byte: float = 0.000002
    message_latent_byte: float = 0.0000002
    agent_call: float = 0.12
    tool_call: float = 0.005
    model_call: float = 0.01
    repeated_agent: float = 0.08
    invalid_action: float = 0.5
    error: float = 0.25
    non_completion: float = 0.75

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"reward weight {name} must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class RoutingRewardBreakdown:
    task_score: float
    task_reward: float
    evidence_quality_gain: float
    structure_quality_gain: float
    message_cost: float
    agent_cost: float
    tool_cost: float
    model_cost: float
    redundancy_cost: float
    invalid_action_cost: float
    error_cost: float
    non_completion_cost: float
    total: float
    schema_version: str = REWARD_SCHEMA

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "schema_version":
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"reward field {name} must be finite")
        recomputed = (
            self.task_reward
            + self.evidence_quality_gain
            + self.structure_quality_gain
            - self.message_cost
            - self.agent_cost
            - self.tool_cost
            - self.model_cost
            - self.redundancy_cost
            - self.invalid_action_cost
            - self.error_cost
            - self.non_completion_cost
        )
        if not math.isclose(self.total, recomputed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reward total does not equal its signed components")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoutingRewardManager:
    """Compute terminal task reward and once-per-episode quality/cost terms."""

    def __init__(self, weights: RoutingRewardWeights | None = None) -> None:
        self.weights = weights or RoutingRewardWeights()

    @staticmethod
    def _evidence_quality(result: MultiAgentRunResult) -> float:
        if not result.workspace.evidence:
            return 0.0
        linked = sum(
            1
            for item in result.workspace.evidence
            if item.spatial_node_ids or item.artifact_refs or item.summary.strip()
        )
        confidence = sum(
            0.5 if item.confidence is None else item.confidence
            for item in result.workspace.evidence
        ) / len(result.workspace.evidence)
        return 0.5 * linked / len(result.workspace.evidence) + 0.5 * confidence

    @staticmethod
    def _structure_quality(result: MultiAgentRunResult) -> float:
        nodes = result.workspace.spatial_nodes
        if not nodes:
            return 0.0
        ids = {item.node_id for item in nodes}
        roots = [item for item in nodes if item.parent_id is None]
        linked = sum(
            1 for item in nodes if item.parent_id is None or item.parent_id in ids
        )
        evidenced = sum(1 for item in nodes if item.evidence_ids)
        root_score = 1.0 if len(roots) == 1 else 0.0
        return (
            root_score
            + linked / len(nodes)
            + evidenced / len(nodes)
        ) / 3.0

    def score(
        self,
        result: MultiAgentRunResult,
        *,
        reference_answer: str,
    ) -> RoutingRewardBreakdown:
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError("reference_answer must be non-empty")
        task_score = score_answer(
            result.task.task_type,
            result.answer or "",
            reference_answer,
        )
        w = self.weights
        usage = result.total_cost
        repeated = len(result.agent_path) - len(set(result.agent_path))
        task_reward = w.task * task_score
        evidence_gain = w.evidence_gain * self._evidence_quality(result)
        structure_gain = w.structure_gain * self._structure_quality(result)
        message_cost = (
            w.message_text_token * usage.message_text_tokens
            + w.message_structure_byte * usage.message_structure_bytes
            + w.message_latent_byte * usage.message_latent_bytes
        )
        agent_cost = w.agent_call * usage.agent_calls
        tool_cost = w.tool_call * usage.tool_calls
        model_cost = w.model_call * usage.model_calls
        redundancy_cost = w.repeated_agent * max(0, repeated)
        invalid_cost = w.invalid_action * usage.invalid_actions
        error_cost = w.error * usage.errors
        non_completion = (
            0.0 if result.stop_reason == StopReason.COMPLETED else w.non_completion
        )
        total = (
            task_reward
            + evidence_gain
            + structure_gain
            - message_cost
            - agent_cost
            - tool_cost
            - model_cost
            - redundancy_cost
            - invalid_cost
            - error_cost
            - non_completion
        )
        return RoutingRewardBreakdown(
            task_score=task_score,
            task_reward=task_reward,
            evidence_quality_gain=evidence_gain,
            structure_quality_gain=structure_gain,
            message_cost=message_cost,
            agent_cost=agent_cost,
            tool_cost=tool_cost,
            model_cost=model_cost,
            redundancy_cost=redundancy_cost,
            invalid_action_cost=invalid_cost,
            error_cost=error_cost,
            non_completion_cost=non_completion,
            total=total,
        )


@dataclass(frozen=True, slots=True)
class RoutingTrainingSample:
    sample_id: str
    task: TaskContext
    reference_answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not isinstance(self.task, TaskContext):
            raise TypeError("task must be a TaskContext")
        if not isinstance(self.reference_answer, str) or not self.reference_answer.strip():
            raise ValueError("reference_answer must be non-empty")
        serialized_task = json.dumps(self.task.to_dict(), ensure_ascii=False, sort_keys=True)
        if self.reference_answer in serialized_task:
            raise ValueError(
                "reference_answer must remain environment-only and must not occur in TaskContext"
            )


class RoutingEpisodeRunner(Protocol):
    def run(
        self,
        policy: LearnedRoutingPolicy,
        sample: RoutingTrainingSample,
        *,
        episode_id: str,
    ) -> MultiAgentRunResult:
        ...

    def expert_fingerprint(self) -> Mapping[str, str]:
        ...


def fingerprint_expert_types(experts: Mapping[str, Any]) -> dict[str, str]:
    """Hash immutable expert code/config, not mutable per-episode counters."""

    result: dict[str, str] = {}
    for expert_id, expert in sorted(experts.items()):
        cls = type(expert)
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            source = f"{cls.__module__}.{cls.__qualname__}"
        instance_values = getattr(expert, "__dict__", {})
        public_config = {
            key: value
            for key, value in instance_values.items()
            if not key.startswith("_")
            and isinstance(value, (str, int, float, bool, type(None), tuple))
        }
        payload = json.dumps(
            {
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "source": source,
                "public_config": public_config,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=repr,
        ).encode("utf-8")
        result[expert_id] = hashlib.sha256(payload).hexdigest()
    return result


def audit_action_causality(
    trace: Sequence[RoutingSample],
    result: MultiAgentRunResult,
) -> dict[str, Any]:
    """Prove sampled actions precede and identify their runtime transitions."""

    transitions = result.transitions
    if len(trace) != len(transitions):
        raise ValueError(
            "sampled action count does not match environment transition count: "
            f"{len(trace)} != {len(transitions)}"
        )
    links: list[dict[str, Any]] = []
    for sample, transition in zip(trace, transitions):
        recorded = transition.decision.get("policy_action_id")
        if recorded != sample.action_id:
            raise ValueError(
                f"sampled action {sample.action_id} does not match transition {recorded}"
            )
        if sample.workspace_version != transition.workspace_version_before:
            raise ValueError("action was not sampled from the transition's pre-state")
        if sample.step != transition.step:
            raise ValueError("action step does not match transition step")
        links.append(
            {
                "step": sample.step,
                "workspace_version_at_sample": sample.workspace_version,
                "action_id": sample.action_id,
                "transition_workspace_version_before": transition.workspace_version_before,
                "transition_workspace_version_after": transition.workspace_version_after,
                "active_expert": transition.active_expert,
            }
        )
    return {
        "all_actions_precede_transitions": True,
        "linked_action_count": len(links),
        "links": links,
    }


@dataclass(frozen=True, slots=True)
class RoutingTrainerConfig:
    epochs: int = 20
    learning_rate: float = 0.08
    baseline_rate: float = 0.1
    entropy_coefficient: float = 0.01
    max_gradient: float = 5.0
    seed: int = 17

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        for name in (
            "learning_rate",
            "baseline_rate",
            "entropy_coefficient",
            "max_gradient",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.learning_rate <= 0 or not 0 < self.baseline_rate <= 1 or self.max_gradient <= 0:
            raise ValueError("invalid routing trainer hyperparameters")


@dataclass(frozen=True, slots=True)
class RoutingTrainingResult:
    initial_parameter_sha256: str
    final_parameter_sha256: str
    expert_fingerprint_before: Mapping[str, str]
    expert_fingerprint_after: Mapping[str, str]
    train_events: tuple[Mapping[str, Any], ...]
    eval_events: tuple[Mapping[str, Any], ...]
    schema_version: str = TRAINING_SCHEMA

    @property
    def experts_frozen(self) -> bool:
        return dict(self.expert_fingerprint_before) == dict(self.expert_fingerprint_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "initial_parameter_sha256": self.initial_parameter_sha256,
            "final_parameter_sha256": self.final_parameter_sha256,
            "parameter_changed": self.initial_parameter_sha256 != self.final_parameter_sha256,
            "expert_fingerprint_before": dict(self.expert_fingerprint_before),
            "expert_fingerprint_after": dict(self.expert_fingerprint_after),
            "experts_frozen": self.experts_frozen,
            "train_events": [dict(item) for item in self.train_events],
            "eval_events": [dict(item) for item in self.eval_events],
        }


class RoutingPolicyTrainer:
    """On-policy trainer whose environment is the real multi-Agent runtime."""

    def __init__(
        self,
        policy: LearnedRoutingPolicy,
        runner: RoutingEpisodeRunner,
        *,
        reward_manager: RoutingRewardManager | None = None,
        config: RoutingTrainerConfig | None = None,
    ) -> None:
        if not isinstance(policy, LearnedRoutingPolicy):
            raise TypeError("policy must be a LearnedRoutingPolicy")
        self.policy = policy
        self.runner = runner
        self.reward_manager = reward_manager or RoutingRewardManager()
        self.config = config or RoutingTrainerConfig()
        self._rng = random.Random(self.config.seed)

    @staticmethod
    def _event(
        *,
        split: str,
        epoch: int | None,
        sample: RoutingTrainingSample,
        result: MultiAgentRunResult,
        trace: Sequence[RoutingSample],
        reward: RoutingRewardBreakdown,
        causality: Mapping[str, Any],
        parameter_before: str,
        parameter_after: str,
        update: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "evors.routing-episode.v1",
            "split": split,
            "epoch": epoch,
            "sample_id": sample.sample_id,
            "task_id": sample.task.task_id,
            "task_type": sample.task.task_type,
            "reference_answer": sample.reference_answer,
            "answer": result.answer,
            "correct": reward.task_score == 1.0,
            "stop_reason": result.stop_reason.value,
            "agent_path": list(result.agent_path),
            "action_trace": [item.to_dict() for item in trace],
            "causality": dict(causality),
            "reward": reward.to_dict(),
            "total_cost": result.total_cost.to_dict(),
            "workspace_version": result.workspace.version,
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
            "optimizer_update": dict(update) if update is not None else None,
            "runtime_result_sha256": hashlib.sha256(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def train_and_evaluate(
        self,
        train_samples: Sequence[RoutingTrainingSample],
        eval_samples: Sequence[RoutingTrainingSample],
    ) -> RoutingTrainingResult:
        if not train_samples or not eval_samples:
            raise ValueError("train_samples and eval_samples must both be non-empty")
        train_ids = {item.sample_id for item in train_samples}
        eval_ids = {item.sample_id for item in eval_samples}
        overlap = train_ids & eval_ids
        if overlap:
            raise ValueError(f"train/eval sample IDs overlap: {sorted(overlap)}")

        expert_before = dict(self.runner.expert_fingerprint())
        initial_digest = self.policy.parameter_digest()
        train_events: list[Mapping[str, Any]] = []
        for epoch in range(self.config.epochs):
            indices = list(range(len(train_samples)))
            self._rng.shuffle(indices)
            for sample_index in indices:
                sample = train_samples[sample_index]
                self.policy.begin_episode(training=True, explore=True)
                before = self.policy.parameter_digest()
                result = self.runner.run(
                    self.policy,
                    sample,
                    episode_id=f"phase5-train-e{epoch:03d}-{sample.sample_id}",
                )
                result.replay()
                trace = self.policy.episode_trace
                causality = audit_action_causality(trace, result)
                reward = self.reward_manager.score(
                    result,
                    reference_answer=sample.reference_answer,
                )
                update = self.policy.update_episode(
                    reward.total,
                    learning_rate=self.config.learning_rate,
                    baseline_rate=self.config.baseline_rate,
                    entropy_coefficient=self.config.entropy_coefficient,
                    max_gradient=self.config.max_gradient,
                )
                after = self.policy.parameter_digest()
                if before == after:
                    raise RuntimeError("optimizer step did not change policy parameters")
                train_events.append(
                    self._event(
                        split="train",
                        epoch=epoch,
                        sample=sample,
                        result=result,
                        trace=trace,
                        reward=reward,
                        causality=causality,
                        parameter_before=before,
                        parameter_after=after,
                        update=update,
                    )
                )

        trained_digest = self.policy.parameter_digest()
        eval_events: list[Mapping[str, Any]] = []
        for sample in eval_samples:
            self.policy.begin_episode(training=False, explore=False)
            before = self.policy.parameter_digest()
            result = self.runner.run(
                self.policy,
                sample,
                episode_id=f"phase5-eval-{sample.sample_id}",
            )
            result.replay()
            trace = self.policy.episode_trace
            causality = audit_action_causality(trace, result)
            reward = self.reward_manager.score(
                result,
                reference_answer=sample.reference_answer,
            )
            after = self.policy.parameter_digest()
            if before != after or after != trained_digest:
                raise RuntimeError("evaluation mutated policy parameters")
            eval_events.append(
                self._event(
                    split="eval",
                    epoch=None,
                    sample=sample,
                    result=result,
                    trace=trace,
                    reward=reward,
                    causality=causality,
                    parameter_before=before,
                    parameter_after=after,
                    update=None,
                )
            )

        expert_after = dict(self.runner.expert_fingerprint())
        result = RoutingTrainingResult(
            initial_parameter_sha256=initial_digest,
            final_parameter_sha256=trained_digest,
            expert_fingerprint_before=expert_before,
            expert_fingerprint_after=expert_after,
            train_events=tuple(train_events),
            eval_events=tuple(eval_events),
        )
        if initial_digest == trained_digest:
            raise RuntimeError("training did not change the policy checkpoint")
        if not result.experts_frozen:
            raise RuntimeError("Cognitive Expert fingerprint changed during controller training")
        return result


def summarize_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(events)
    if not items:
        return {"episodes": 0, "accuracy": 0.0, "mean_reward": 0.0, "paths": {}}
    paths: dict[str, int] = {}
    for item in items:
        key = "->".join(item["agent_path"])
        paths[key] = paths.get(key, 0) + 1
    return {
        "episodes": len(items),
        "accuracy": sum(bool(item["correct"]) for item in items) / len(items),
        "mean_reward": sum(float(item["reward"]["total"]) for item in items) / len(items),
        "mean_agent_calls": sum(int(item["total_cost"]["agent_calls"]) for item in items) / len(items),
        "paths": paths,
    }


__all__ = [
    "REWARD_SCHEMA",
    "TRAINING_SCHEMA",
    "RoutingEpisodeRunner",
    "RoutingPolicyTrainer",
    "RoutingRewardBreakdown",
    "RoutingRewardManager",
    "RoutingRewardWeights",
    "RoutingTrainerConfig",
    "RoutingTrainingResult",
    "RoutingTrainingSample",
    "audit_action_causality",
    "fingerprint_expert_types",
    "score_answer",
    "summarize_events",
]
