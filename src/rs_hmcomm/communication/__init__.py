"""Reference-routed communication with scoped Workspace observations."""

from .protocol import AccessGrant, Inbox, MessageEnvelope, MessageModality
from .routing import ContextRouter, ObservationBuilder

__all__ = [
    "AccessGrant",
    "ContextRouter",
    "Inbox",
    "MessageEnvelope",
    "MessageModality",
    "ObservationBuilder",
]
