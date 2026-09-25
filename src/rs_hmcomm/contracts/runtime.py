"""Phase 0 runtime, budget, cost, and event contracts.

The event stream is the canonical cost source.  ``CostLedger`` is an
immutable view that can be reconstructed from those events; runtimes should
not maintain an independent set of counters beside the trace.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import json
import math
from typing import Any, Iterable, Mapping

from ._validation import json_object_copy, normalize_json_object, strict_dict


# A recursive JSON type alias is awkward to express portably on Python 3.10.
# Runtime safety comes from ``validate_json_value`` below.
JsonValue = Any


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return converted


def validate_json_value(value: Any, field_name: str = "value") -> JsonValue:
    """Validate that a value can safely cross a JSON contract boundary."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [validate_json_value(item, field_name) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} contains a non-string mapping key")
            result[key] = validate_json_value(item, field_name)
        return result
    raise ValueError(
        f"{field_name} must contain JSON values only; got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class CostVector:
    """Raw, additive counters used for budgeting and matched-cost evaluation."""

    agent_calls: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    message_text_tokens: int = 0
    message_structure_bytes: int = 0
    message_latent_bytes: int = 0
    billed_compute_ms: float = 0.0
    errors: int = 0
    retries: int = 0
    invalid_actions: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "billed_compute_ms":
                object.__setattr__(self, item.name, _non_negative_float(value, item.name))
            else:
                _non_negative_int(value, item.name)

    def __add__(self, other: "CostVector") -> "CostVector":
        if not isinstance(other, CostVector):
            return NotImplemented
        values = {
            item.name: getattr(self, item.name) + getattr(other, item.name)
            for item in fields(self)
        }
        return CostVector(**values)

    @property
    def model_tokens(self) -> int:
        return self.model_input_tokens + self.model_output_tokens

    @property
    def communication_bytes(self) -> int:
        return self.message_structure_bytes + self.message_latent_bytes

    def to_dict(self) -> dict[str, int | float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostVector":
        if not isinstance(data, Mapping):
            raise ValueError("cost must be a mapping")
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown cost fields: {sorted(unknown)}")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Hard upper limits. ``None`` means that dimension is not constrained."""

    max_agent_calls: int | None = None
    max_tool_calls: int | None = None
    max_model_calls: int | None = None
    max_model_input_tokens: int | None = None
    max_model_output_tokens: int | None = None
    max_message_text_tokens: int | None = None
    max_message_structure_bytes: int | None = None
    max_message_latent_bytes: int | None = None
    max_billed_compute_ms: float | None = None
    max_wall_time_ms: float | None = None
    max_errors: int | None = None
    max_retries: int | None = None
    max_invalid_actions: int | None = None

    _COST_FIELD_BY_LIMIT = {
        "max_agent_calls": "agent_calls",
        "max_tool_calls": "tool_calls",
        "max_model_calls": "model_calls",
        "max_model_input_tokens": "model_input_tokens",
        "max_model_output_tokens": "model_output_tokens",
        "max_message_text_tokens": "message_text_tokens",
        "max_message_structure_bytes": "message_structure_bytes",
        "max_message_latent_bytes": "message_latent_bytes",
        "max_billed_compute_ms": "billed_compute_ms",
        "max_errors": "errors",
        "max_retries": "retries",
        "max_invalid_actions": "invalid_actions",
    }

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            if item.name in {"max_billed_compute_ms", "max_wall_time_ms"}:
                object.__setattr__(self, item.name, _non_negative_float(value, item.name))
            else:
                _non_negative_int(value, item.name)

    def violations(
        self,
        usage: CostVector,
        *,
        wall_time_ms: float | None = None,
    ) -> tuple[str, ...]:
        violated: list[str] = []
        for limit_name, cost_name in self._COST_FIELD_BY_LIMIT.items():
            limit = getattr(self, limit_name)
            if limit is not None and getattr(usage, cost_name) > limit:
                violated.append(cost_name)
        if wall_time_ms is not None:
            wall_time = _non_negative_float(wall_time_ms, "wall_time_ms")
            if self.max_wall_time_ms is not None and wall_time > self.max_wall_time_ms:
                violated.append("wall_time_ms")
        return tuple(violated)

    def allows(
        self,
        usage: CostVector,
        delta: CostVector | None = None,
        *,
        wall_time_ms: float | None = None,
    ) -> bool:
        projected = usage if delta is None else usage + delta
        return not self.violations(projected, wall_time_ms=wall_time_ms)

    def require_within(
        self,
        usage: CostVector,
        delta: CostVector | None = None,
        *,
        wall_time_ms: float | None = None,
    ) -> None:
        projected = usage if delta is None else usage + delta
        violated = self.violations(projected, wall_time_ms=wall_time_ms)
        if violated:
            raise BudgetExceeded(violated, projected, wall_time_ms=wall_time_ms)

    def to_dict(self) -> dict[str, int | float | None]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetLimits":
        if not isinstance(data, Mapping):
            raise ValueError("budget limits must be a mapping")
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown budget fields: {sorted(unknown)}")
        return cls(**dict(data))


class BudgetExceeded(RuntimeError):
    def __init__(
        self,
        dimensions: tuple[str, ...],
        projected: CostVector,
        *,
        wall_time_ms: float | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.projected = projected
        self.wall_time_ms = wall_time_ms
        super().__init__(f"budget exceeded: {', '.join(dimensions)}")


@dataclass(frozen=True, slots=True)
class CostEntry:
    label: str
    delta: CostVector
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.label, "label")
        if self.source_event_id is not None:
            _non_empty(self.source_event_id, "source_event_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "delta": self.delta.to_dict(),
            "source_event_id": self.source_event_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostEntry":
        values = strict_dict(
            data,
            cls.__name__,
            required={"label", "delta", "source_event_id"},
        )
        return cls(
            label=values["label"],
            delta=CostVector.from_dict(values["delta"]),
            source_event_id=values["source_event_id"],
        )


@dataclass(frozen=True, slots=True)
class CostLedger:
    """Immutable list of cost deltas, normally reconstructed from events."""

    entries: tuple[CostEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        source_ids = [
            entry.source_event_id
            for entry in self.entries
            if entry.source_event_id is not None
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_event_id values must be unique within a ledger")

    @property
    def total(self) -> CostVector:
        total = CostVector()
        for entry in self.entries:
            total = total + entry.delta
        return total

    def append(self, entry: CostEntry) -> "CostLedger":
        return CostLedger(self.entries + (entry,))

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostLedger":
        values = strict_dict(data, cls.__name__, required={"entries"})
        raw_entries = values["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("entries must be a list")
        return cls(tuple(CostEntry.from_dict(item) for item in raw_entries))


class StopReason(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_STEPS = "max_steps"
    INVALID_ACTION = "invalid_action"
    TOOL_ERROR = "tool_error"
    BACKEND_ERROR = "backend_error"
    TIMEOUT = "timeout"
    NO_PROGRESS = "no_progress"
    ABORTED = "aborted"


class EventType(str, Enum):
    EPISODE_STARTED = "episode_started"
    OBSERVATION_BUILT = "observation_built"
    AGENT_DISPATCHED = "agent_dispatched"
    MODEL_CALLED = "model_called"
    AGENT_ACTED = "agent_acted"
    TOOL_CALLED = "tool_called"
    CANDIDATE_PROPOSED = "candidate_proposed"
    CANDIDATE_EVALUATED = "candidate_evaluated"
    CANDIDATE_SELECTED = "candidate_selected"
    RECORD_ADDED = "record_added"
    MESSAGE_ROUTED = "message_routed"
    WARNING = "warning"
    ERROR = "error"
    EPISODE_FINISHED = "episode_finished"


_ALLOWED_COST_FIELDS: dict[EventType, frozenset[str]] = {
    EventType.AGENT_DISPATCHED: frozenset({"agent_calls"}),
    EventType.MODEL_CALLED: frozenset(
        {"model_calls", "model_input_tokens", "model_output_tokens", "billed_compute_ms"}
    ),
    EventType.TOOL_CALLED: frozenset({"tool_calls", "billed_compute_ms"}),
    EventType.MESSAGE_ROUTED: frozenset(
        {
            "message_text_tokens",
            "message_structure_bytes",
            "message_latent_bytes",
            "billed_compute_ms",
        }
    ),
    EventType.WARNING: frozenset({"retries"}),
    EventType.ERROR: frozenset({"errors", "invalid_actions"}),
}


_CALL_COUNTER_BY_EVENT: dict[EventType, str] = {
    EventType.AGENT_DISPATCHED: "agent_calls",
    EventType.MODEL_CALLED: "model_calls",
    EventType.TOOL_CALLED: "tool_calls",
}


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    episode_id: str
    event_id: str
    seq: int
    event_type: EventType
    actor: str
    action: Mapping[str, JsonValue]
    result: Mapping[str, JsonValue]
    cost_delta: CostVector = CostVector()
    elapsed_ms: float | None = None
    schema_version: str = "evors.event.v1"

    def __post_init__(self) -> None:
        _non_empty(self.episode_id, "episode_id")
        _non_empty(self.event_id, "event_id")
        _non_negative_int(self.seq, "seq")
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        _non_empty(self.actor, "actor")
        if self.schema_version != "evors.event.v1":
            raise ValueError("unsupported event schema_version")
        if not isinstance(self.action, Mapping) or not isinstance(self.result, Mapping):
            raise ValueError("action and result must be mappings")
        action = dict(self.action)
        if self.event_type in _CALL_COUNTER_BY_EVENT and "call_id" not in action:
            # Legacy emitters did not carry call_id for dispatch/model events.
            # event_id is already unique inside the episode and is therefore a
            # safe canonical identifier for that one physical call.
            action["call_id"] = self.event_id
        object.__setattr__(self, "action", normalize_json_object(action, "action"))
        object.__setattr__(self, "result", normalize_json_object(self.result, "result"))
        if not isinstance(self.cost_delta, CostVector):
            raise TypeError("cost_delta must be a CostVector")
        nonzero_cost_fields = {
            item.name
            for item in fields(self.cost_delta)
            if getattr(self.cost_delta, item.name) != 0
        }
        allowed_cost_fields = _ALLOWED_COST_FIELDS.get(self.event_type, frozenset())
        disallowed = nonzero_cost_fields - allowed_cost_fields
        if disallowed:
            raise ValueError(
                f"{self.event_type.value} cannot own cost fields: {sorted(disallowed)}"
            )
        counter_field = _CALL_COUNTER_BY_EVENT.get(self.event_type)
        if counter_field is not None and getattr(self.cost_delta, counter_field) != 1:
            raise ValueError(
                f"{self.event_type.value} must own exactly one {counter_field}"
            )
        if self.cost_delta.invalid_actions > 1:
            raise ValueError("one event can own at most one invalid_actions count")
        if self.elapsed_ms is not None:
            object.__setattr__(
                self,
                "elapsed_ms",
                _non_negative_float(self.elapsed_ms, "elapsed_ms"),
            )
        self._validate_event_payload()

    def _validate_event_payload(self) -> None:
        def require(mapping: Mapping[str, Any], names: set[str], side: str) -> None:
            missing = names - set(mapping)
            if missing:
                raise ValueError(
                    f"{self.event_type.value} {side} is missing fields: {sorted(missing)}"
                )

        if self.event_type == EventType.EPISODE_STARTED:
            require(self.action, {"task_id"}, "action")
            require(self.result, {"observation_id"}, "result")
            _non_empty(self.action["task_id"], "task_id")
            _non_empty(self.result["observation_id"], "observation_id")
            if "budget_limits" in self.result:
                BudgetLimits.from_dict(self.result["budget_limits"])
        elif self.event_type == EventType.AGENT_DISPATCHED:
            require(self.action, {"agent_id", "assignment_id", "call_id"}, "action")
            _non_empty(self.action["agent_id"], "agent_id")
            _non_empty(self.action["assignment_id"], "assignment_id")
            _non_empty(self.action["call_id"], "call_id")
        elif self.event_type == EventType.MODEL_CALLED:
            require(self.action, {"model_id", "call_id"}, "action")
            require(self.result, {"token_accounting"}, "result")
            _non_empty(self.action["model_id"], "model_id")
            _non_empty(self.action["call_id"], "call_id")
            if self.result["token_accounting"] not in {
                "measured",
                "estimated",
                "unavailable",
            }:
                raise ValueError("model_called has an invalid token_accounting value")
        elif self.event_type == EventType.TOOL_CALLED:
            require(self.action, {"tool_name", "call_id"}, "action")
            _non_empty(self.action["tool_name"], "tool_name")
            _non_empty(self.action["call_id"], "call_id")
        elif self.event_type == EventType.CANDIDATE_PROPOSED:
            require(self.action, {"candidate_id", "generation"}, "action")
            require(
                self.result,
                {
                    "parent_candidate_id",
                    "variation_operator",
                    "payload_sha256",
                    "scope",
                },
                "result",
            )
            _non_empty(self.action["candidate_id"], "candidate_id")
            _non_negative_int(self.action["generation"], "generation")
            parent_id = self.result["parent_candidate_id"]
            if parent_id is not None:
                _non_empty(parent_id, "parent_candidate_id")
            _non_empty(self.result["variation_operator"], "variation_operator")
            _non_empty(self.result["payload_sha256"], "payload_sha256")
            if self.result["scope"] != "private_agent_scratchpad":
                raise ValueError("candidate_proposed must remain private Agent scratch")
        elif self.event_type == EventType.CANDIDATE_EVALUATED:
            require(self.action, {"candidate_id", "evaluator"}, "action")
            require(self.result, {"score", "components"}, "result")
            _non_empty(self.action["candidate_id"], "candidate_id")
            _non_empty(self.action["evaluator"], "evaluator")
            score = self.result["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError("candidate_evaluated score must be finite")
            if not isinstance(self.result["components"], Mapping):
                raise ValueError("candidate_evaluated components must be an object")
        elif self.event_type == EventType.CANDIDATE_SELECTED:
            require(self.action, {"generation", "strategy"}, "action")
            require(
                self.result,
                {"selected_candidate_id", "candidate_scores", "fitness_gain"},
                "result",
            )
            _non_negative_int(self.action["generation"], "generation")
            _non_empty(self.action["strategy"], "strategy")
            _non_empty(
                self.result["selected_candidate_id"],
                "selected_candidate_id",
            )
            if not isinstance(self.result["candidate_scores"], Mapping):
                raise ValueError("candidate_selected candidate_scores must be an object")
            _non_negative_float(self.result["fitness_gain"], "fitness_gain")
        elif self.event_type == EventType.RECORD_ADDED:
            require(self.result, {"records"}, "result")
            records = self.result["records"]
            if not isinstance(records, tuple) or not records:
                raise ValueError("record_added result.records must be a non-empty list")
            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise ValueError(f"record_added records[{index}] must be an object")
                require(record, {"record_id", "record_type"}, f"records[{index}]")
                _non_empty(record["record_id"], f"records[{index}].record_id")
                _non_empty(record["record_type"], f"records[{index}].record_type")
        elif self.event_type == EventType.EPISODE_FINISHED:
            require(self.result, {"stop_reason"}, "result")
            try:
                StopReason(self.result["stop_reason"])
            except (TypeError, ValueError) as exc:
                raise ValueError("episode_finished has an invalid stop_reason") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "event_id": self.event_id,
            "seq": self.seq,
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": json_object_copy(self.action),
            "result": json_object_copy(self.result),
            "cost_delta": self.cost_delta.to_dict(),
            "elapsed_ms": self.elapsed_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeEvent":
        values = strict_dict(
            data,
            cls.__name__,
            required={
                "schema_version",
                "episode_id",
                "event_id",
                "seq",
                "event_type",
                "actor",
                "action",
                "result",
                "cost_delta",
                "elapsed_ms",
            },
        )
        return cls(
            schema_version=values["schema_version"],
            episode_id=values["episode_id"],
            event_id=values["event_id"],
            seq=values["seq"],
            event_type=EventType(values["event_type"]),
            actor=values["actor"],
            action=values["action"],
            result=values["result"],
            cost_delta=CostVector.from_dict(values["cost_delta"]),
            elapsed_ms=values["elapsed_ms"],
        )

    @classmethod
    def from_json(cls, raw: str) -> "EpisodeEvent":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("event JSON must contain an object")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    episode_id: str
    events: tuple[EpisodeEvent, ...] = ()
    budget_limits: BudgetLimits | None = None

    def __post_init__(self) -> None:
        _non_empty(self.episode_id, "episode_id")
        object.__setattr__(self, "events", tuple(self.events))
        if self.budget_limits is not None and not isinstance(
            self.budget_limits, BudgetLimits
        ):
            raise TypeError("budget_limits must be a BudgetLimits or None")
        elapsed_values: list[float] = []
        for expected_seq, event in enumerate(self.events):
            if event.episode_id != self.episode_id:
                raise ValueError("event episode_id does not match trace")
            if event.seq != expected_seq:
                raise ValueError(
                    f"event sequence must be contiguous from zero; expected {expected_seq}, "
                    f"got {event.seq}"
                )
            if event.elapsed_ms is not None:
                elapsed_values.append(event.elapsed_ms)
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a trace")
        if elapsed_values and len(elapsed_values) != len(self.events):
            raise ValueError("elapsed_ms must be present on every event or none")
        if any(right < left for left, right in zip(elapsed_values, elapsed_values[1:])):
            raise ValueError("event elapsed_ms values must be monotonic")
        # A completed physical call can exceed a token/time limit even when it
        # passed its pre-flight gate.  Keep the measured event and declared
        # limits together so replay can report the violation instead of
        # dropping either the cost or the budget contract.

    def append(
        self,
        event: EpisodeEvent,
        *,
        budget_limits: BudgetLimits | None = None,
        enforce_budget: bool = True,
    ) -> "EpisodeTrace":
        if event.episode_id != self.episode_id:
            raise ValueError("event episode_id does not match trace")
        if event.seq != len(self.events):
            raise ValueError(f"next event seq must be {len(self.events)}")
        if any(previous.event_id == event.event_id for previous in self.events):
            raise ValueError(f"duplicate event_id: {event.event_id}")
        effective_budget = self.budget_limits
        if budget_limits is not None:
            if effective_budget is not None and effective_budget != budget_limits:
                raise ValueError("budget_limits cannot change within an episode trace")
            effective_budget = budget_limits
        projected_wall_time: float | None = None
        if event.elapsed_ms is not None:
            if self.events and self.events[0].elapsed_ms is not None:
                projected_wall_time = event.elapsed_ms - self.events[0].elapsed_ms
            elif not self.events:
                projected_wall_time = 0.0
        if effective_budget is not None and enforce_budget:
            # The event is the canonical cost source.  Gate the actual delta
            # being appended rather than a caller-supplied estimate.
            effective_budget.require_within(
                self.total_cost,
                event.cost_delta,
                wall_time_ms=projected_wall_time,
            )
        return EpisodeTrace(
            episode_id=self.episode_id,
            events=self.events + (event,),
            budget_limits=effective_budget,
        )

    @property
    def total_cost(self) -> CostVector:
        total = CostVector()
        for event in self.events:
            total = total + event.cost_delta
        return total

    @property
    def wall_time_ms(self) -> float | None:
        if not self.events or self.events[0].elapsed_ms is None:
            return None
        return self.events[-1].elapsed_ms - self.events[0].elapsed_ms

    @property
    def ledger(self) -> CostLedger:
        return CostLedger(
            tuple(
                CostEntry(
                    label=event.event_type.value,
                    delta=event.cost_delta,
                    source_event_id=event.event_id,
                )
                for event in self.events
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "events": [event.to_dict() for event in self.events],
            "budget_limits": (
                self.budget_limits.to_dict()
                if self.budget_limits is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeTrace":
        values = strict_dict(
            data,
            cls.__name__,
            required={"episode_id", "events"},
            optional={"budget_limits"},
        )
        raw_events = values["events"]
        if not isinstance(raw_events, list):
            raise ValueError("events must be a list")
        return cls(
            episode_id=values["episode_id"],
            events=tuple(EpisodeEvent.from_dict(item) for item in raw_events),
            budget_limits=(
                BudgetLimits.from_dict(values["budget_limits"])
                if values.get("budget_limits") is not None
                else None
            ),
        )

    @classmethod
    def replay(
        cls,
        events: Iterable[EpisodeEvent],
        *,
        budget_limits: BudgetLimits | None = None,
    ) -> "EpisodeTrace":
        iterator = iter(events)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("cannot replay an empty event stream") from exc
        trace = cls(first.episode_id, budget_limits=budget_limits).append(
            first,
            enforce_budget=False,
        )
        for event in iterator:
            trace = trace.append(event, enforce_budget=False)
        return trace

    def validate_complete(self) -> "ReplayState":
        """Replay lifecycle and record effects for a finished episode."""

        if not self.events:
            raise ValueError("a complete trace cannot be empty")
        if self.events[0].event_type != EventType.EPISODE_STARTED:
            raise ValueError("the first event must be episode_started")
        if self.events[-1].event_type != EventType.EPISODE_FINISHED:
            raise ValueError("the last event must be episode_finished")
        if sum(event.event_type == EventType.EPISODE_STARTED for event in self.events) != 1:
            raise ValueError("a complete trace must contain exactly one episode_started")
        if sum(event.event_type == EventType.EPISODE_FINISHED for event in self.events) != 1:
            raise ValueError("a complete trace must contain exactly one episode_finished")

        call_owners: dict[str, str] = {}
        for event in self.events:
            if event.event_type not in _CALL_COUNTER_BY_EVENT:
                continue
            call_id = event.action["call_id"]
            previous = call_owners.get(call_id)
            if previous is not None:
                raise ValueError(
                    f"call_id {call_id!r} is reused by events {previous!r} "
                    f"and {event.event_id!r}"
                )
            call_owners[call_id] = event.event_id

        records: dict[str, str] = {}
        agent_path: list[str] = []
        for event in self.events:
            if event.event_type == EventType.AGENT_DISPATCHED:
                agent_path.append(event.action["agent_id"])
            elif event.event_type == EventType.RECORD_ADDED:
                for record in event.result["records"]:
                    record_id = record["record_id"]
                    if record_id in records:
                        raise ValueError(f"record {record_id!r} is added more than once")
                    records[record_id] = record["record_type"]

        finish = self.events[-1]
        stop_reason = StopReason(finish.result["stop_reason"])
        budget_violations = (
            self.budget_limits.violations(
                self.total_cost,
                wall_time_ms=self.wall_time_ms,
            )
            if self.budget_limits is not None
            else ()
        )
        if budget_violations and stop_reason != StopReason.BUDGET_EXHAUSTED:
            raise ValueError(
                "a trace that exceeds its declared budget must finish with "
                "budget_exhausted"
            )
        answer_ref = finish.action.get("answer_candidate_ref")
        if stop_reason == StopReason.COMPLETED:
            if not isinstance(answer_ref, str) or not answer_ref.strip():
                raise ValueError("completed traces require answer_candidate_ref")
            if records.get(answer_ref) != "answer_candidate":
                raise ValueError("finished answer_candidate_ref does not resolve")
        elif answer_ref is not None:
            raise ValueError("non-completed traces must not select an answer candidate")

        return ReplayState(
            episode_id=self.episode_id,
            task_id=self.events[0].action["task_id"],
            record_types=records,
            agent_path=tuple(agent_path),
            answer_candidate_ref=answer_ref,
            stop_reason=stop_reason,
            total_cost=self.total_cost,
            wall_time_ms=self.wall_time_ms,
            budget_violations=budget_violations,
        )


@dataclass(frozen=True)
class ReplayState:
    episode_id: str
    task_id: str
    record_types: Mapping[str, str]
    agent_path: tuple[str, ...]
    answer_candidate_ref: str | None
    stop_reason: StopReason
    total_cost: CostVector
    wall_time_ms: float | None
    budget_violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_types", normalize_json_object(self.record_types, "record_types"))
        object.__setattr__(self, "agent_path", tuple(self.agent_path))
        object.__setattr__(self, "budget_violations", tuple(self.budget_violations))
