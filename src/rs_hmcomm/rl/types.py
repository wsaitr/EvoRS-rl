from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from rs_hmcomm.core import NodeLevel, MessageModality

class TaskAction(str, Enum):
    INSPECT = "inspect"
    VERIFY = "verify"
    AGGREGATE = "aggregate"
    ANSWER = "answer"
    STOP = "stop"

@dataclass
class CommunicationAction:
    task_action: TaskAction
    recipient: str
    spatial_level: NodeLevel
    modality: tuple[MessageModality, ...]
    payload_node_ids: tuple[str, ...] = ()
