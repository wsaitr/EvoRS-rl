"""Validation boundary around a routing policy."""
from __future__ import annotations

from rs_hmcomm.contracts import StopReason

from .contracts import (
    AbortDecision,
    ControllerDecision,
    ControllerObservation,
    DispatchDecision,
    FinishDecision,
    RoutingPolicy,
)


EXPERT_IDS = frozenset({"visual", "reasoning", "verification", "knowledge"})


class EvolutionController:
    """Control-layer policy host; deliberately not a Cognitive Expert."""

    def __init__(self, policy: RoutingPolicy) -> None:
        decide = getattr(policy, "decide", None)
        if not callable(decide):
            raise TypeError("policy must implement decide(observation)")
        self.policy = policy

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        if not isinstance(observation, ControllerObservation):
            raise TypeError("observation must be a ControllerObservation")
        decision = self.policy.decide(observation)
        if not isinstance(
            decision,
            (DispatchDecision, FinishDecision, AbortDecision),
        ):
            raise TypeError("routing policy returned an unsupported decision")

        if observation.step >= observation.max_steps:
            if not (
                isinstance(decision, AbortDecision)
                and decision.stop_reason == StopReason.MAX_STEPS
            ):
                raise ValueError("policy must abort with max_steps at the step limit")
            return decision

        if isinstance(decision, DispatchDecision):
            if decision.expert_id not in EXPERT_IDS:
                raise ValueError(f"unknown expert_id: {decision.expert_id}")
            requested = {
                *decision.context_spec.payload_refs,
                *decision.context_spec.focus_node_ids,
            }
            unknown = requested - set(observation.workspace.record_ids)
            if unknown:
                raise ValueError(
                    f"DispatchDecision references missing workspace records: {sorted(unknown)}"
                )
            if len(requested) > decision.context_spec.max_items:
                raise ValueError("DispatchDecision exceeds context_spec.max_items")
            for record_id in decision.context_spec.focus_node_ids:
                record = next(
                    item
                    for item in observation.workspace.records
                    if item.record_id == record_id
                )
                if record.record_type != "spatial_node":
                    raise ValueError("focus_node_ids must reference spatial nodes")
            if "latent" in decision.context_spec.modalities:
                incompatible = tuple(
                    item.record_id
                    for item in observation.workspace.records
                    if item.record_id in requested
                    and item.record_type == "artifact"
                    and decision.expert_id not in item.compatible_consumers
                )
                if incompatible:
                    raise ValueError(
                        f"latent artifacts are incompatible with {decision.expert_id}: "
                        f"{list(incompatible)}"
                    )
            return decision

        if isinstance(decision, FinishDecision):
            candidates = {
                item.record_id: item
                for item in observation.workspace.records
                if item.record_type == "answer_candidate"
            }
            if decision.answer_candidate_ref not in candidates:
                raise ValueError("FinishDecision references a missing answer candidate")
            if candidates[decision.answer_candidate_ref].status != "verified":
                raise ValueError("FinishDecision requires a verified answer candidate")
            return decision

        return decision


# Backward-compatible import name used by earlier experiment scripts.  The
# canonical paper/code name is EvolutionController.
EvoRSController = EvolutionController


__all__ = ["EXPERT_IDS", "EvoRSController", "EvolutionController"]
