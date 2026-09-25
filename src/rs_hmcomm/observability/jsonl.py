"""Append-only JSONL storage for canonical episode events."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rs_hmcomm.contracts.runtime import BudgetLimits, EpisodeEvent, EpisodeTrace


class EventJsonlLogger:
    """Write one event per line so partial episodes remain inspectable."""

    def __init__(self, output_path: str | Path, *, append: bool = True) -> None:
        self.output_path = Path(output_path)
        if self.output_path.exists() and self.output_path.is_dir():
            raise ValueError("event log output_path must be a file, not a directory")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        # event_id is local to an episode; a JSONL file may interleave many
        # episodes that legitimately reuse the same local event names.
        self._event_keys: set[tuple[str, str]] = set()
        self._next_seq_by_episode: dict[str, int] = {}
        self._count = 0
        if append and self.output_path.exists():
            if self.output_path.stat().st_size:
                with self.output_path.open("rb") as handle:
                    handle.seek(-1, 2)
                    if handle.read(1) != b"\n":
                        raise ValueError("existing event log does not end with a newline")
            existing = read_events(self.output_path)
            for event in existing:
                event_key = (event.episode_id, event.event_id)
                if event_key in self._event_keys:
                    raise ValueError(
                        "existing event log has duplicate event key: "
                        f"{event_key!r}"
                    )
                expected = self._next_seq_by_episode.get(event.episode_id, 0)
                if event.seq != expected:
                    raise ValueError(
                        f"existing event sequence for {event.episode_id} expected {expected}, "
                        f"got {event.seq}"
                    )
                self._event_keys.add(event_key)
                self._next_seq_by_episode[event.episode_id] = expected + 1
            self._count = len(existing)

    def write_event(self, event: EpisodeEvent) -> None:
        event_key = (event.episode_id, event.event_id)
        if event_key in self._event_keys:
            raise ValueError(f"duplicate event key: {event_key!r}")
        expected = self._next_seq_by_episode.get(event.episode_id, 0)
        if event.seq != expected:
            raise ValueError(
                f"next event seq for {event.episode_id} must be {expected}, got {event.seq}"
            )
        with self.output_path.open(self._mode, encoding="utf-8", newline="\n") as handle:
            handle.write(event.to_json())
            handle.write("\n")
        self._mode = "a"
        self._event_keys.add(event_key)
        self._next_seq_by_episode[event.episode_id] = expected + 1
        self._count += 1

    def write_trace(self, trace: EpisodeTrace) -> None:
        if trace.budget_limits is not None:
            if not trace.events or trace.events[0].event_type.value != "episode_started":
                raise ValueError("a budgeted trace must begin with episode_started")
            declared = trace.events[0].result.get("budget_limits")
            if declared is None or BudgetLimits.from_dict(declared) != trace.budget_limits:
                raise ValueError(
                    "trace budget_limits must match episode_started.result.budget_limits"
                )
        for event in trace.events:
            self.write_event(event)

    @property
    def count(self) -> int:
        return self._count


def read_events(path: str | Path) -> tuple[EpisodeEvent, ...]:
    events: list[EpisodeEvent] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("line must contain a JSON object")
                events.append(EpisodeEvent.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid event JSONL at line {line_number}: {exc}") from exc
    return tuple(events)


def load_event_traces(path: str | Path) -> tuple[EpisodeTrace, ...]:
    """Group interleaved events by episode, enforcing each local sequence."""

    traces_by_id: dict[str, EpisodeTrace] = {}
    order: list[str] = []
    for event in read_events(path):
        if event.episode_id not in traces_by_id:
            declared = event.result.get("budget_limits")
            budget = BudgetLimits.from_dict(declared) if declared is not None else None
            traces_by_id[event.episode_id] = EpisodeTrace(
                event.episode_id,
                budget_limits=budget,
            )
            order.append(event.episode_id)
        traces_by_id[event.episode_id] = traces_by_id[event.episode_id].append(
            event,
            enforce_budget=False,
        )
    return tuple(traces_by_id[episode_id] for episode_id in order)
