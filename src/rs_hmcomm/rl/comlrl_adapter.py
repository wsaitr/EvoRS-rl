from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class CoMLRLTrajectoryAdapter:
    upstream_commit: str = "UNPINNED"

    def ensure_installed(self) -> None:
        try:
            import comlrl  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "CoMLRL is not installed. Use `pip install comlrl` or clone a pinned upstream revision."
            ) from e

    def episode_to_training_record(self, episode: Any, answer: str) -> dict[str, Any]:
        return {
            "answer": answer,
            "tree": episode.tree.to_dict(),
            "message_stats": episode.bus.stats(),
            "agent_outputs": episode.outputs,
            "communication_metadata": [
                {
                    "sender": m.sender,
                    "receiver": m.receiver,
                    "level": m.spatial_level.value,
                    "modalities": [x.value for x in m.modalities],
                    "node_ids": list(m.node_ids),
                }
                for m in episode.bus.messages
            ],
        }
