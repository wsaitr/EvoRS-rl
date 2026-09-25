"""Trainable centralized routing policy for EvoRS Phase 5.

The policy is intentionally small: a masked linear softmax controller over a
versioned, compact ``ControllerObservation`` feature vector. Training code may
update the weights with policy gradients, while inference only needs this file
and a JSON checkpoint. Cognitive Expert parameters are never owned here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from rs_hmcomm.contracts import RecordStatus, StopReason

from .contracts import (
    AbortDecision,
    ContextSpec,
    ControllerDecision,
    ControllerObservation,
    ControllerWorkspaceSummary,
    DispatchDecision,
    FinishDecision,
)
from .policies import RuleRoutingPolicy, _context_for


CHECKPOINT_SCHEMA = "evors.routing-policy.v1"
FEATURE_SCHEMA = "evors.controller-features.v1"

ACTION_IDS = (
    "dispatch_visual",
    "dispatch_knowledge",
    "dispatch_reasoning_text_structure",
    "dispatch_reasoning_hybrid",
    "dispatch_verification",
    "finish",
    "abort",
)

FEATURE_NAMES = (
    "bias",
    "step_fraction",
    "task_quantity",
    "task_existence",
    "task_classification",
    "question_count_signal",
    "question_existence_signal",
    "question_classification_signal",
    "has_visual_evidence",
    "has_knowledge_evidence",
    "has_proposed_candidate",
    "has_verified_candidate",
    "has_compatible_latent",
    "used_visual",
    "used_knowledge",
    "used_reasoning",
    "used_verification",
    "message_count_scaled",
    "agent_calls_scaled",
    "model_calls_scaled",
    "capability_requests_scaled",
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _softmax(logits: Sequence[float], mask: Sequence[bool]) -> tuple[float, ...]:
    valid = [index for index, allowed in enumerate(mask) if allowed]
    if not valid:
        raise ValueError("routing action mask contains no valid action")
    maximum = max(logits[index] for index in valid)
    exps = {
        index: math.exp(_clamp(logits[index] - maximum, -60.0, 60.0))
        for index in valid
    }
    denominator = sum(exps.values())
    return tuple(
        exps.get(index, 0.0) / denominator
        for index in range(len(logits))
    )


def _sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative and probability > 0.0:
            return index
    return max(
        (index for index, probability in enumerate(probabilities) if probability > 0.0),
        key=lambda index: probabilities[index],
    )


def _has_record(
    summary: ControllerWorkspaceSummary,
    *,
    record_type: str,
    created_by: str | None = None,
    status: str | None = None,
) -> bool:
    return any(
        item.record_type == record_type
        and (created_by is None or item.created_by == created_by)
        and (status is None or item.status == status)
        for item in summary.records
    )


def _compatible_latent(summary: ControllerWorkspaceSummary) -> bool:
    return any(
        item.record_type == "artifact"
        and item.subtype == "latent"
        and "reasoning" in item.compatible_consumers
        for item in summary.records
    )


def encode_observation(observation: ControllerObservation) -> tuple[float, ...]:
    """Encode only the compact controller view; evaluation labels are absent."""

    summary = observation.workspace
    question = observation.task.question.casefold()
    task_type = observation.task.task_type.casefold()
    quantity = any(token in task_type for token in ("quantity", "count"))
    existence = any(token in task_type for token in ("exist", "presence"))
    classification = any(
        token in task_type for token in ("class", "type", "category", "scene")
    )
    count_signal = any(
        token in question
        for token in ("how many", "count", "number of", "多少", "几", "统计")
    )
    existence_signal = any(
        token in question
        for token in ("is there", "are there", "any ", "是否", "有没有", "存在")
    )
    classification_signal = any(
        token in question
        for token in ("what type", "classify", "category", "类型", "类别", "判断")
    )
    path = frozenset(observation.agent_path)
    cost = summary.total_cost
    values = (
        1.0,
        observation.step / max(1, observation.max_steps),
        float(quantity),
        float(existence),
        float(classification),
        float(count_signal),
        float(existence_signal),
        float(classification_signal),
        float(_has_record(summary, record_type="evidence", created_by="visual")),
        float(_has_record(summary, record_type="evidence", created_by="knowledge")),
        float(
            _has_record(
                summary,
                record_type="answer_candidate",
                status=RecordStatus.PROPOSED.value,
            )
        ),
        float(bool(summary.verified_candidate_ids)),
        float(_compatible_latent(summary)),
        float("visual" in path),
        float("knowledge" in path),
        float("reasoning" in path),
        float("verification" in path),
        min(1.0, summary.message_count / 8.0),
        min(1.0, cost.agent_calls / max(1, observation.max_steps)),
        min(1.0, cost.model_calls / 16.0),
        min(1.0, len(observation.capability_requests) / 4.0),
    )
    if len(values) != len(FEATURE_NAMES):  # pragma: no cover - schema guard
        raise RuntimeError("feature encoder and feature schema have diverged")
    return values


def valid_action_mask(observation: ControllerObservation) -> tuple[bool, ...]:
    """Return the state-dependent safety mask for ``ACTION_IDS``."""

    summary = observation.workspace
    if observation.step >= observation.max_steps:
        return tuple(action == "abort" for action in ACTION_IDS)
    has_visual = _has_record(summary, record_type="evidence", created_by="visual")
    has_knowledge = _has_record(summary, record_type="evidence", created_by="knowledge")
    has_proposed = _has_record(
        summary,
        record_type="answer_candidate",
        status=RecordStatus.PROPOSED.value,
    )
    has_verified = bool(summary.verified_candidate_ids)
    has_claim = _has_record(summary, record_type="claim")
    path = observation.agent_path
    mask_by_id = {
        "dispatch_visual": not has_visual and "visual" not in path,
        "dispatch_knowledge": (
            has_visual
            and not has_knowledge
            and not has_proposed
            and "knowledge" not in path
        ),
        "dispatch_reasoning_text_structure": (
            has_visual and not has_proposed and not has_verified and "reasoning" not in path
        ),
        "dispatch_reasoning_hybrid": (
            has_visual
            and _compatible_latent(summary)
            and not has_proposed
            and not has_verified
            and "reasoning" not in path
        ),
        "dispatch_verification": (
            has_proposed and has_claim and not has_verified and "verification" not in path
        ),
        "finish": has_verified,
        "abort": observation.step > 0 and not has_verified,
    }
    mask = tuple(mask_by_id[action] for action in ACTION_IDS)
    if not any(mask):
        return tuple(action == "abort" for action in ACTION_IDS)
    return mask


def _reasoning_text_structure_context(
    summary: ControllerWorkspaceSummary,
) -> ContextSpec:
    base = _context_for("reasoning", summary)
    artifacts = {
        item.record_id for item in summary.records if item.record_type == "artifact"
    }
    return ContextSpec(
        modalities=("text", "structure"),
        payload_refs=tuple(item for item in base.payload_refs if item not in artifacts),
        focus_node_ids=base.focus_node_ids,
        spatial_level=base.spatial_level,
        max_items=base.max_items,
    )


def action_to_decision(
    action_id: str,
    observation: ControllerObservation,
) -> ControllerDecision:
    """Decode one sampled action into the existing control contract."""

    if action_id not in ACTION_IDS:
        raise ValueError(f"unknown routing action: {action_id}")
    mask = valid_action_mask(observation)
    if not mask[ACTION_IDS.index(action_id)]:
        raise ValueError(f"routing action is invalid in the current state: {action_id}")
    if action_id == "finish":
        return FinishDecision(
            observation.workspace.verified_candidate_ids[-1],
            policy_action_id=action_id,
        )
    if action_id == "abort":
        reason = (
            StopReason.MAX_STEPS
            if observation.step >= observation.max_steps
            else StopReason.NO_PROGRESS
        )
        return AbortDecision(reason, policy_action_id=action_id)

    expert_by_action = {
        "dispatch_visual": "visual",
        "dispatch_knowledge": "knowledge",
        "dispatch_reasoning_text_structure": "reasoning",
        "dispatch_reasoning_hybrid": "reasoning",
        "dispatch_verification": "verification",
    }
    expert_id = expert_by_action[action_id]
    if action_id == "dispatch_reasoning_text_structure":
        context = _reasoning_text_structure_context(observation.workspace)
    else:
        context = _context_for(expert_id, observation.workspace)
    objectives = {
        "visual": "Gather question-relevant visual evidence with private active perception",
        "knowledge": "Provide task-relevant remote-sensing domain guidance",
        "reasoning": "Reason over the policy-selected scoped evidence",
        "verification": "Verify the proposed answer against disclosed evidence",
    }
    return DispatchDecision(
        expert_id,
        objectives[expert_id],
        context_spec=context,
        policy_action_id=action_id,
    )


@dataclass(frozen=True, slots=True)
class RoutingSample:
    """One action sampled before its corresponding environment transition."""

    step: int
    workspace_version: int
    features: tuple[float, ...]
    action_mask: tuple[bool, ...]
    action_index: int
    probability: float
    entropy: float

    @property
    def action_id(self) -> str:
        return ACTION_IDS[self.action_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "workspace_version": self.workspace_version,
            "features": list(self.features),
            "action_mask": list(self.action_mask),
            "action_index": self.action_index,
            "action_id": self.action_id,
            "probability": self.probability,
            "entropy": self.entropy,
        }


class LearnedRoutingPolicy:
    """Masked linear policy with safe untrained fallback and JSON checkpoints."""

    def __init__(
        self,
        *,
        weights: Sequence[Sequence[float]] | None = None,
        biases: Sequence[float] | None = None,
        trained_steps: int = 0,
        baseline: float = 0.0,
        context_baselines: Mapping[str, float] | None = None,
        fallback: RuleRoutingPolicy | None = None,
        seed: int = 0,
    ) -> None:
        self.fallback = fallback or RuleRoutingPolicy()
        self.weights = self._normalize_weights(weights)
        self.biases = self._normalize_biases(biases)
        if isinstance(trained_steps, bool) or not isinstance(trained_steps, int) or trained_steps < 0:
            raise ValueError("trained_steps must be a non-negative integer")
        if not math.isfinite(float(baseline)):
            raise ValueError("baseline must be finite")
        self.trained_steps = trained_steps
        self.baseline = float(baseline)
        self.context_baselines = {
            str(key): float(value) for key, value in (context_baselines or {}).items()
        }
        if any(not math.isfinite(value) for value in self.context_baselines.values()):
            raise ValueError("context baselines must be finite")
        self._rng = random.Random(seed)
        self._training = False
        self._explore = False
        self._episode_trace: list[RoutingSample] = []
        self._last_update_l2 = 0.0

    @staticmethod
    def _normalize_weights(
        weights: Sequence[Sequence[float]] | None,
    ) -> list[list[float]]:
        if weights is None:
            return [[0.0 for _ in FEATURE_NAMES] for _ in ACTION_IDS]
        result = [[float(value) for value in row] for row in weights]
        if len(result) != len(ACTION_IDS) or any(
            len(row) != len(FEATURE_NAMES) for row in result
        ):
            raise ValueError("routing weight matrix shape does not match schemas")
        if any(not math.isfinite(value) for row in result for value in row):
            raise ValueError("routing weights must be finite")
        return result

    @staticmethod
    def _normalize_biases(biases: Sequence[float] | None) -> list[float]:
        result = [0.0 for _ in ACTION_IDS] if biases is None else [float(v) for v in biases]
        if len(result) != len(ACTION_IDS):
            raise ValueError("routing bias vector shape does not match action schema")
        if any(not math.isfinite(value) for value in result):
            raise ValueError("routing biases must be finite")
        return result

    @property
    def episode_trace(self) -> tuple[RoutingSample, ...]:
        return tuple(self._episode_trace)

    @property
    def last_update_l2(self) -> float:
        return self._last_update_l2

    def begin_episode(self, *, training: bool, explore: bool = True) -> None:
        self._training = bool(training)
        self._explore = bool(explore and training)
        self._episode_trace = []
        self._last_update_l2 = 0.0

    def _logits(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            self.biases[action] + sum(
                weight * feature
                for weight, feature in zip(self.weights[action], features)
            )
            for action in range(len(ACTION_IDS))
        )

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        if self.trained_steps == 0 and not self._training:
            return self.fallback.decide(observation)

        features = encode_observation(observation)
        mask = valid_action_mask(observation)
        probabilities = _softmax(self._logits(features), mask)
        if self._explore:
            action_index = _sample_index(probabilities, self._rng)
        else:
            action_index = max(
                (index for index, valid in enumerate(mask) if valid),
                key=lambda index: (probabilities[index], -index),
            )
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities
            if probability > 0.0
        )
        sample = RoutingSample(
            step=observation.step,
            workspace_version=observation.workspace.version,
            features=features,
            action_mask=mask,
            action_index=action_index,
            probability=probabilities[action_index],
            entropy=entropy,
        )
        self._episode_trace.append(sample)
        return action_to_decision(sample.action_id, observation)

    def update_episode(
        self,
        reward: float,
        *,
        learning_rate: float = 0.05,
        baseline_rate: float = 0.1,
        entropy_coefficient: float = 0.01,
        max_gradient: float = 5.0,
    ) -> dict[str, float | int]:
        """Apply one on-policy REINFORCE update to the collected episode."""

        if not self._training:
            raise RuntimeError("begin_episode(training=True) is required before update")
        numeric = (reward, learning_rate, baseline_rate, entropy_coefficient, max_gradient)
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("policy update values must be finite")
        if learning_rate <= 0 or not 0.0 < baseline_rate <= 1.0 or max_gradient <= 0:
            raise ValueError("invalid policy-gradient hyperparameters")
        if not self._episode_trace:
            raise RuntimeError("cannot update from an empty routing trace")

        first_features = self._episode_trace[0].features
        if first_features[2] > 0.5:
            context_key = "quantity"
        elif first_features[3] > 0.5:
            context_key = "existence"
        elif first_features[4] > 0.5:
            context_key = "classification"
        else:
            context_key = "other"
        context_baseline = self.context_baselines.get(context_key, 0.0)
        advantage = float(reward) - context_baseline
        update_sq = 0.0
        for sample in self._episode_trace:
            probabilities = _softmax(self._logits(sample.features), sample.action_mask)
            entropy = -sum(
                probability * math.log(max(probability, 1e-12))
                for probability in probabilities
                if probability > 0.0
            )
            for action_index, probability in enumerate(probabilities):
                if not sample.action_mask[action_index]:
                    continue
                policy_gradient = (
                    (1.0 if action_index == sample.action_index else 0.0)
                    - probability
                ) * advantage
                entropy_gradient = -probability * (
                    math.log(max(probability, 1e-12)) + entropy
                )
                logit_gradient = _clamp(
                    policy_gradient + entropy_coefficient * entropy_gradient,
                    -max_gradient,
                    max_gradient,
                )
                bias_delta = learning_rate * logit_gradient
                self.biases[action_index] += bias_delta
                update_sq += bias_delta * bias_delta
                for feature_index, feature in enumerate(sample.features):
                    delta = learning_rate * logit_gradient * feature
                    self.weights[action_index][feature_index] += delta
                    update_sq += delta * delta

        self.baseline = (
            (1.0 - baseline_rate) * self.baseline + baseline_rate * float(reward)
        )
        self.context_baselines[context_key] = (
            (1.0 - baseline_rate) * context_baseline + baseline_rate * float(reward)
        )
        self.trained_steps += 1
        self._last_update_l2 = math.sqrt(update_sq)
        if not math.isfinite(self._last_update_l2) or self._last_update_l2 <= 0.0:
            raise RuntimeError("policy optimizer produced a zero or non-finite update")
        return {
            "reward": float(reward),
            "advantage": advantage,
            "baseline": self.baseline,
            "context_baseline": self.context_baselines[context_key],
            "update_l2": self._last_update_l2,
            "sample_count": len(self._episode_trace),
            "trained_steps": self.trained_steps,
        }

    def parameter_digest(self) -> str:
        payload = {
            "weights": self.weights,
            "biases": self.biases,
            "trained_steps": self.trained_steps,
            "baseline": self.baseline,
            "context_baselines": self.context_baselines,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def checkpoint_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "action_ids": list(ACTION_IDS),
            "weights": self.weights,
            "biases": self.biases,
            "trained_steps": self.trained_steps,
            "baseline": self.baseline,
            "context_baselines": self.context_baselines,
        }
        payload["parameter_sha256"] = self.parameter_digest()
        return payload

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.checkpoint_dict(), indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_checkpoint(
        cls,
        path_or_payload: str | Path | Mapping[str, Any],
        *,
        seed: int = 0,
    ) -> "LearnedRoutingPolicy":
        if isinstance(path_or_payload, Mapping):
            data = dict(path_or_payload)
        else:
            data = json.loads(Path(path_or_payload).read_text(encoding="utf-8"))
        if data.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported routing checkpoint schema")
        if data.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("unsupported controller feature schema")
        if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("checkpoint feature names do not match runtime")
        if tuple(data.get("action_ids", ())) != ACTION_IDS:
            raise ValueError("checkpoint action IDs do not match runtime")
        policy = cls(
            weights=data["weights"],
            biases=data["biases"],
            trained_steps=int(data["trained_steps"]),
            baseline=float(data["baseline"]),
            context_baselines=data.get("context_baselines", {}),
            seed=seed,
        )
        expected = data.get("parameter_sha256")
        if expected != policy.parameter_digest():
            raise ValueError("routing checkpoint parameter checksum mismatch")
        return policy


__all__ = [
    "ACTION_IDS",
    "CHECKPOINT_SCHEMA",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "LearnedRoutingPolicy",
    "RoutingSample",
    "action_to_decision",
    "encode_observation",
    "valid_action_mask",
]
