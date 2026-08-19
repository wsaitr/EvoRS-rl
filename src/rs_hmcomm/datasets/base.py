from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class RSSample:
    sample_id: str
    image: Any
    question: str
    answer: str
    task_type: str = "vqa"
    bbox: Optional[tuple[float, float, float, float]] = None
    metadata: dict[str, Any] | None = None
