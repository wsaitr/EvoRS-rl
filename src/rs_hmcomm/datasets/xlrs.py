from __future__ import annotations
from typing import Iterator
from .base import RSSample

def stream_xlrs_lite(split: str = "train", limit: int | None = None) -> Iterator[RSSample]:
    from datasets import load_dataset
    ds = load_dataset("initiacms/XLRS-Bench-lite", split=split, streaming=True)
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        image = row.get("image")
        q = row.get("question") or row.get("prompt") or ""
        a = row.get("answer") or row.get("label") or ""
        yield RSSample(str(row.get("id", idx)), image, str(q), str(a), metadata={"raw_keys": list(row.keys())})
