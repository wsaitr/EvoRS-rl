"""Reference-only routed communication records for Phase 4."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Iterable, Mapping

from rs_hmcomm.contracts import CostVector
from rs_hmcomm.contracts._validation import (
    normalize_id_tuple,
    require_id,
    require_step,
    require_text,
    strict_dict,
)


class MessageModality(str, Enum):
    TEXT = "text"
    STRUCTURE = "structure"
    LATENT = "latent"


def _modalities(
    values: Iterable[MessageModality | str],
) -> tuple[MessageModality, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("modalities must be an iterable of modality values")
    try:
        normalized = tuple(
            value if isinstance(value, MessageModality) else MessageModality(value)
            for value in values
        )
    except TypeError as exc:
        raise TypeError("modalities must be an iterable of modality values") from exc
    except ValueError as exc:
        raise ValueError("modalities may contain only text, structure, and latent") from exc
    if not normalized:
        raise ValueError("modalities must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("modalities must not contain duplicates")
    order = {item: index for index, item in enumerate(MessageModality)}
    return tuple(sorted(normalized, key=order.__getitem__))


def _workspace_refs(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("workspace_refs must be an iterable of record IDs")
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise TypeError("workspace_refs must be an iterable of record IDs") from exc
    checked = normalize_id_tuple(
        normalized,
        "workspace_refs",
        require_nonempty=True,
    )
    return tuple(sorted(checked))


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    """A logical disclosure: refs and modalities, never copied payloads."""

    message_id: str
    sender: str
    receiver: str
    modalities: tuple[MessageModality, ...]
    workspace_refs: tuple[str, ...]
    scope: str
    created_step: int

    def __post_init__(self) -> None:
        require_id(self.message_id, "message_id")
        require_id(self.sender, "sender")
        require_id(self.receiver, "receiver")
        object.__setattr__(self, "modalities", _modalities(self.modalities))
        object.__setattr__(self, "workspace_refs", _workspace_refs(self.workspace_refs))
        require_text(self.scope, "scope")
        require_step(self.created_step, "created_step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "modalities": [item.value for item in self.modalities],
            "workspace_refs": list(self.workspace_refs),
            "scope": self.scope,
            "created_step": self.created_step,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MessageEnvelope":
        values = strict_dict(
            data,
            cls.__name__,
            required={
                "message_id",
                "sender",
                "receiver",
                "modalities",
                "workspace_refs",
                "scope",
                "created_step",
            },
        )
        return cls(
            message_id=values["message_id"],
            sender=values["sender"],
            receiver=values["receiver"],
            modalities=values["modalities"],
            workspace_refs=values["workspace_refs"],
            scope=values["scope"],
            created_step=values["created_step"],
        )

    @classmethod
    def from_json(cls, raw: str) -> "MessageEnvelope":
        if not isinstance(raw, str):
            raise TypeError("MessageEnvelope JSON must be a string")
        data = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
        if not isinstance(data, Mapping):
            raise ValueError("MessageEnvelope JSON must contain an object")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class AccessGrant:
    """Authorization issued with one routed envelope."""

    grant_id: str
    message_id: str
    grantor: str
    grantee: str
    workspace_refs: tuple[str, ...]
    created_step: int
    cost: CostVector = field(default_factory=CostVector)

    def __post_init__(self) -> None:
        require_id(self.grant_id, "grant_id")
        require_id(self.message_id, "message_id")
        require_id(self.grantor, "grantor")
        require_id(self.grantee, "grantee")
        object.__setattr__(self, "workspace_refs", _workspace_refs(self.workspace_refs))
        require_step(self.created_step, "created_step")
        if not isinstance(self.cost, CostVector):
            raise TypeError("cost must be a CostVector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "message_id": self.message_id,
            "grantor": self.grantor,
            "grantee": self.grantee,
            "workspace_refs": list(self.workspace_refs),
            "created_step": self.created_step,
            "cost": self.cost.to_dict(),
        }


class Inbox:
    """Insertion-ordered replayable message collection."""

    def __init__(self) -> None:
        self._messages: list[MessageEnvelope] = []
        self._message_ids: set[str] = set()

    def append(self, message: MessageEnvelope) -> MessageEnvelope:
        if not isinstance(message, MessageEnvelope):
            raise TypeError("message must be a MessageEnvelope")
        if message.message_id in self._message_ids:
            raise ValueError(f"duplicate message_id: {message.message_id}")
        self._messages.append(message)
        self._message_ids.add(message.message_id)
        return message

    def remove(self, message_id: str) -> MessageEnvelope:
        require_id(message_id, "message_id")
        for index, message in enumerate(self._messages):
            if message.message_id == message_id:
                removed = self._messages.pop(index)
                self._message_ids.remove(message_id)
                return removed
        raise KeyError(f"unknown message_id: {message_id}")

    def by_recipient(self, recipient: str) -> tuple[MessageEnvelope, ...]:
        require_id(recipient, "recipient")
        return tuple(
            message for message in self._messages if message.receiver == recipient
        )

    @property
    def messages(self) -> tuple[MessageEnvelope, ...]:
        return tuple(self._messages)

    def to_dict(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self._messages]

    @classmethod
    def replay(
        cls,
        messages: Iterable[MessageEnvelope | Mapping[str, Any]],
    ) -> "Inbox":
        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be an iterable of envelopes")
        inbox = cls()
        for item in messages:
            envelope = (
                item
                if isinstance(item, MessageEnvelope)
                else MessageEnvelope.from_dict(item)
            )
            inbox.append(envelope)
        return inbox

    def __len__(self) -> int:
        return len(self._messages)


__all__ = ["AccessGrant", "Inbox", "MessageEnvelope", "MessageModality"]
