"""Experience-based routing policy learned from past episodes.

``LearnedRoutingPolicy`` implements H3/H5's dynamic routing baseline: it
records (question features, chosen expert) outcomes from completed runs and
routes new questions by nearest-neighbour similarity over token features,
falling back to the rule policy for unseen regions.  It is deliberately
simple, deterministic given its memory, and free of any neural training —
the point is a *learned* controller whose behaviour differs from the fixed
and rule baselines while remaining fully auditable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from rs_hmcomm.contracts import RecordStatus, StopReason

from .contracts import (
    AbortDecision,
    ControllerDecision,
    ControllerObservation,
    DispatchDecision,
    FinishDecision,
)
from .policies import RuleRoutingPolicy


_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.UNICODE)


def _tokenize(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.casefold()))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b)


class LearnedRoutingPolicy:
    """Nearest-neighbour expert sequencing over remembered episodes.

    Memory entries map a question token vector to the expert sequence that
    produced an accepted answer.  Routing picks the most similar memory; if
    no memory exceeds ``min_similarity`` the rule policy answers instead.
    """

    def __init__(
        self,
        *,
        min_similarity: float = 0.15,
        max_memories: int = 512,
        fallback: RuleRoutingPolicy | None = None,
    ) -> None:
        if isinstance(min_similarity, bool) or not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be within [0, 1]")
        self.min_similarity = min_similarity
        self.max_memories = max_memories
        self.fallback = fallback or RuleRoutingPolicy()
        self._memories: list[dict[str, Any]] = []

    # -- learning ----------------------------------------------------------

    def remember(
        self,
        question: str,
        sequence: tuple[str, ...],
        *,
        correct: bool,
    ) -> None:
        """Store one episode outcome.  Only correct episodes teach routing."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not sequence or any(not item for item in sequence):
            raise ValueError("sequence must be a non-empty tuple of expert ids")
        if len(self._memories) >= self.max_memories:
            self._memories.pop(0)
        if not correct:
            # Negative outcomes do not add memories but shrink nothing; the
            # caller may re-route such questions through other sequences.
            return
        self._memories.append(
            {
                "tokens": _tokenize(question),
                "sequence": tuple(sequence),
                "question": question,
            }
        )

    @property
    def memory_size(self) -> int:
        return len(self._memories)

    # -- RoutingPolicy -----------------------------------------------------

    def decide(self, observation: ControllerObservation) -> ControllerDecision:
        if observation.step >= observation.max_steps:
            return AbortDecision(StopReason.MAX_STEPS)

        sequence = self._route(observation.task.question)
        next_index = 0
        for executed in observation.agent_path:
            if next_index < len(sequence) and executed == sequence[next_index]:
                next_index += 1
        if next_index < len(sequence):
            expert_id = sequence[next_index]
            return DispatchDecision(expert_id, f"learned route step {next_index + 1}/{len(sequence)}")
        return self._finish_or_abort(observation)

    def _finish_or_abort(self, observation: ControllerObservation) -> ControllerDecision:
        verified = tuple(
            candidate
            for candidate in observation.workspace.answer_candidates
            if candidate.status == RecordStatus.VERIFIED
        )
        if not verified:
            return AbortDecision(StopReason.NO_PROGRESS)
        return FinishDecision(verified[-1].candidate_id)

    def _route(self, question: str) -> tuple[str, ...]:
        if not self._memories:
            return self.fallback._sequence_for(question)
        query = _tokenize(question)
        best_similarity = -1.0
        best_sequence: tuple[str, ...] | None = None
        votes: dict[tuple[str, ...], float] = {}
        for memory in self._memories:
            similarity = _cosine(query, memory["tokens"])
            if similarity >= self.min_similarity:
                votes[memory["sequence"]] = votes.get(memory["sequence"], 0.0) + similarity
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_sequence = memory["sequence"]
        if best_sequence is None:
            return self.fallback._sequence_for(question)
        # Weighted vote across similar memories breaks ties deterministically.
        return max(votes.items(), key=lambda item: (item[1], item[0]))[0]


__all__ = ["LearnedRoutingPolicy"]
