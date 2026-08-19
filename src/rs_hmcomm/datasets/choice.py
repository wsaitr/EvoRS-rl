from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
from PIL import Image
from .base import RSSample

def iter_choice_subset(root: str | Path) -> Iterator[RSSample]:
    root = Path(root)
    for json_path in root.rglob("*.json"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("data", data.get("samples", []))
        if not isinstance(items, list):
            continue
        task = json_path.stem
        image_dir = json_path.parent / "images"
        for i, row in enumerate(items):
            image_name = row.get("image") or row.get("image_path") or row.get("img")
            if not image_name:
                continue
            image_path = image_dir / Path(image_name).name
            if not image_path.exists():
                continue
            question = row.get("question") or row.get("prompt") or ""
            answer = row.get("answer") or row.get("label") or ""
            yield RSSample(
                f"{task}:{i}",
                Image.open(image_path).convert("RGB"),
                str(question),
                str(answer),
                task_type=task,
                metadata={"source_json": str(json_path)},
            )
