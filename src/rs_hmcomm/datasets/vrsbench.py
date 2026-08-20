from __future__ import annotations
from typing import Iterator
import logging
from pathlib import Path

from .base import RSSample

logger = logging.getLogger(__name__)


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


def stream_vrsbench_from_obs(obs_prefix: str = "vrsbench/", cache_dir: str = "./data/cache",
                              split: str = "train", limit: int | None = None) -> Iterator[RSSample]:
    """Stream VRSBench samples from OBS with local caching."""
    from rs_hmcomm.data.obs_loader import OBSDataLoader, OBSConfig
    from rs_hmcomm.data.cache import DataCache
    from PIL import Image
    import json

    config = OBSConfig.from_env()
    loader = OBSDataLoader(config, cache_dir=cache_dir)
    cache = DataCache(cache_dir=cache_dir)

    # Download manifest first
    manifest_key = f"{obs_prefix}{split}/manifest.json"
    try:
        manifest_path = loader.download(manifest_key)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load VRSBench manifest from OBS: {e}")

    items = manifest if isinstance(manifest, list) else manifest.get("samples", [])

    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            break

        # Download image if not cached
        img_key = item.get("image_key", f"{obs_prefix}{split}/images/{item.get('image', '')}")
        cache_key = f"vrsbench/{split}/{img_key}"

        if cache.has(cache_key):
            img_path = cache.get(cache_key)
        else:
            try:
                img_path = loader.download(img_key)
                cache.put(cache_key, img_path)
            except Exception as e:
                logger.warning(f"Failed to download {img_key}: {e}")
                continue

        image = Image.open(img_path).convert("RGB")
        question = item.get("question", "")
        answer = item.get("answer", "")
        sid = item.get("id", str(idx))

        yield RSSample(sid, image, str(question), str(answer), metadata={"obs_key": img_key})
