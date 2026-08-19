from __future__ import annotations
from typing import Iterator
from .base import RSSample

def stream_vrsbench(split: str = "train", limit: int | None = None) -> Iterator[RSSample]:
    # Best-effort adapter: inspect a real row and pin schema before formal experiments.
    from datasets import load_dataset
    ds = load_dataset("xiang709/VRSBench", split=split, streaming=True)
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        image = row.get("image")
        q = row.get("question") or row.get("prompt") or ""
        a = row.get("answer") or row.get("response") or ""
        sid = str(row.get("id", idx))
        yield RSSample(sid, image, str(q), str(a), metadata={"raw_keys": list(row.keys())})
