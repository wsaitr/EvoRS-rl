"""Version-tolerant verl boundary for EvoRS routing trajectories.

The Phase 5 controller is an independent discrete policy, so its optimizer does
not require verl.  This adapter prepares the same environment trajectory for a
future token-policy AgentLoop and exposes a custom reward function compatible
with verl's documented ``compute_score`` contract.  It never reports a dry-run
as framework training.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rs_hmcomm.control import LearnedRoutingPolicy

from .routing import (
    RoutingEpisodeRunner,
    RoutingRewardManager,
    RoutingTrainingSample,
    audit_action_causality,
)


VERL_RECORD_SCHEMA = "evors.verl-routing-record.v1"


@dataclass(frozen=True, slots=True)
class EvoRSAgentLoopResult:
    """Framework-neutral output of one real EvoRS environment rollout."""

    sample_id: str
    answer: str | None
    action_trace: tuple[Mapping[str, Any], ...]
    runtime_result: Mapping[str, Any]
    reward: Mapping[str, Any]
    causality: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "answer": self.answer,
            "action_trace": [dict(item) for item in self.action_trace],
            "runtime_result": dict(self.runtime_result),
            "reward": dict(self.reward),
            "causality": dict(self.causality),
        }


class EvoRSAgentLoop:
    """Run policy -> LangGraph environment -> reward in verl-compatible order."""

    def __init__(
        self,
        policy: LearnedRoutingPolicy,
        runner: RoutingEpisodeRunner,
        *,
        reward_manager: RoutingRewardManager | None = None,
    ) -> None:
        self.policy = policy
        self.runner = runner
        self.reward_manager = reward_manager or RoutingRewardManager()

    def run(
        self,
        sample: RoutingTrainingSample,
        *,
        episode_id: str,
        training: bool = False,
        explore: bool = False,
    ) -> EvoRSAgentLoopResult:
        self.policy.begin_episode(training=training, explore=explore)
        result = self.runner.run(self.policy, sample, episode_id=episode_id)
        result.replay()
        trace = self.policy.episode_trace
        causality = audit_action_causality(trace, result)
        reward = self.reward_manager.score(
            result,
            reference_answer=sample.reference_answer,
        )
        return EvoRSAgentLoopResult(
            sample_id=sample.sample_id,
            answer=result.answer,
            action_trace=tuple(item.to_dict() for item in trace),
            runtime_result=result.to_dict(),
            reward=reward.to_dict(),
            causality=causality,
        )


class VerlRoutingAdapter:
    """Serialize EvoRS samples/rollouts into verl RLHF dataset records."""

    @staticmethod
    def training_record(
        sample: RoutingTrainingSample,
        rollout: EvoRSAgentLoopResult | None = None,
    ) -> dict[str, Any]:
        extra_info: dict[str, Any] = {
            "schema_version": VERL_RECORD_SCHEMA,
            "sample_id": sample.sample_id,
            "task": sample.task.to_dict(),
        }
        if rollout is not None:
            extra_info["evors_rollout"] = rollout.to_dict()
        return {
            "data_source": "evors_phase5_routing",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "Choose valid EvoRS routing actions from the compact controller "
                        "observation. Do not invent visual facts or final answers."
                    ),
                },
                {"role": "user", "content": sample.task.question},
            ],
            "ability": "remote_sensing_dynamic_routing",
            "reward_model": {
                "style": "rule",
                "ground_truth": sample.reference_answer,
            },
            "extra_info": extra_info,
            "agent_name": "evors_routing_agent_loop",
        }

    @classmethod
    def write_jsonl(
        cls,
        samples: Sequence[RoutingTrainingSample],
        path: str | Path,
        *,
        rollouts: Mapping[str, EvoRSAgentLoopResult] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for sample in samples:
            rollout = None if rollouts is None else rollouts.get(sample.sample_id)
            lines.append(
                json.dumps(
                    cls.training_record(sample, rollout),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any],
    **_: Any,
) -> float:
    """Custom verl reward function for an environment-executed trajectory.

    The score is accepted only from a causally audited EvoRS rollout.  It is not
    inferred from model-authored text, preventing a policy from fabricating its
    own cost or reward fields.
    """

    del solution_str, ground_truth
    if data_source != "evors_phase5_routing":
        raise ValueError("unexpected data_source for EvoRS routing reward")
    if not isinstance(extra_info, Mapping):
        raise TypeError("extra_info must be a mapping")
    rollout = extra_info.get("evors_rollout")
    if not isinstance(rollout, Mapping):
        raise ValueError("EvoRS reward requires an executed evors_rollout")
    causality = rollout.get("causality")
    if not isinstance(causality, Mapping) or causality.get(
        "all_actions_precede_transitions"
    ) is not True:
        raise ValueError("EvoRS rollout has not passed action-causality audit")
    reward = rollout.get("reward")
    if not isinstance(reward, Mapping):
        raise ValueError("EvoRS rollout is missing reward breakdown")
    total = float(reward.get("total"))
    if not math.isfinite(total):
        raise ValueError("EvoRS rollout reward must be finite")
    return total


def verl_environment_status() -> dict[str, Any]:
    """Report import/version status without turning absence into fake success."""

    try:
        import verl  # type: ignore
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return {
            "installed": False,
            "version": None,
            "error_type": type(exc).__name__,
        }
    return {
        "installed": True,
        "version": getattr(verl, "__version__", "unknown"),
        "error_type": None,
    }


__all__ = [
    "EvoRSAgentLoop",
    "EvoRSAgentLoopResult",
    "VERL_RECORD_SCHEMA",
    "VerlRoutingAdapter",
    "compute_score",
    "verl_environment_status",
]
