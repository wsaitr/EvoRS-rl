"""Paired Direct-vs-Single-Agent perception gate on VRSBench samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rs_hmcomm.agents.active_perception import (  # noqa: E402
    ActivePerceptionAgent,
    direct_prompt,
)
from rs_hmcomm.backends.base import BackendRequest, VisualInput  # noqa: E402
from rs_hmcomm.backends.openai_compatible import (  # noqa: E402
    OpenAICompatibleVLMBackend,
)


DEFAULT_MANIFEST = Path("configs/experiments/vrsbench20_manifest.jsonl")
DEFAULT_MODEL_CONFIG = Path("configs/models/qwen38_27b_q4_local.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        required = {
            "sample_id",
            "image_ref",
            "task_type",
            "question",
            "reference_answer",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"sample is missing fields: {sorted(missing)}")
        samples.append(item)
    if not samples:
        raise ValueError("manifest has no samples")
    return samples


def _score(task_type: str, answer: str, reference: str) -> bool:
    if task_type == "object_quantity":
        match = re.search(r"\d+", answer.strip())
        return bool(match) and match.group(0) == reference.strip()
    match = re.search(r"\b(yes|no)\b", answer.casefold())
    return bool(match) and match.group(1) == reference.casefold().strip()


def _backend(config_path: Path, seed: int) -> OpenAICompatibleVLMBackend:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    identity = config["identity"]
    server = config["server"]
    return OpenAICompatibleVLMBackend(
        identity["api_alias"],
        base_url=f"http://{server['host']}:{server['port']}/v1",
        api_key="local",
        timeout=180.0,
        max_tokens=256,
        extra_body={"temperature": 0.0, "seed": seed},
        enable_thinking=False,
        preserve_thinking=False,
    )


def _direct(
    backend: OpenAICompatibleVLMBackend,
    image: Image.Image,
    question: str,
    task_type: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = backend.generate_turn(
        BackendRequest(
            prompt=direct_prompt(question, task_type),
            visuals=(VisualInput("input_image", "whole_image", image),),
        )
    )
    return {
        "answer": output.text.strip(),
        "model_calls": 1,
        "tool_calls": 0,
        "input_tokens": output.usage.input_tokens or 0,
        "output_tokens": output.usage.output_tokens or 0,
        "wall_time_ms": (time.perf_counter() - started) * 1000.0,
        "model_ids": [output.model_id],
        "observations": [
            {
                "view": "whole_image",
                "answer": output.text.strip(),
                "input_tokens": output.usage.input_tokens or 0,
                "output_tokens": output.usage.output_tokens or 0,
                "latency_ms": output.usage.latency_ms,
                "model_id": output.model_id,
            }
        ],
        "decision_rule": "single_whole_image_call",
    }


def _agent(
    agent: ActivePerceptionAgent,
    image: Image.Image,
    question: str,
    task_type: str,
) -> dict[str, Any]:
    result = agent.run(image, question, task_type)
    return {
        "answer": result.answer,
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_time_ms": result.wall_time_ms,
        "model_ids": sorted({item.model_id for item in result.calls}),
        "observations": [item.to_dict() for item in result.calls],
        "decision_rule": result.decision_rule,
    }


def run(
    manifest: Path,
    model_config: Path,
    output_dir: Path,
    *,
    offset: int = 0,
    limit: int | None = None,
    seed: int = 20260831,
    grid_sizes: tuple[int, ...] = (),
    tile_scan_size: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    summary_path = output_dir / "summary.json"
    if not overwrite and (events_path.exists() or summary_path.exists()):
        raise FileExistsError("output already exists; pass --overwrite")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    samples = _load_samples(manifest)[offset:]
    if limit is not None:
        samples = samples[:limit]
    backend = _backend(model_config, seed)
    agent = ActivePerceptionAgent(
        backend,
        grid_sizes=grid_sizes,
        tile_scan_size=tile_scan_size,
    )
    records: list[dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            image = Image.open(sample["image_ref"]).convert("RGB")
            for arm in ("direct", "agent"):
                record: dict[str, Any] = {
                    "sample_id": sample["sample_id"],
                    "task_type": sample["task_type"],
                    "question": sample["question"],
                    "reference_answer": sample["reference_answer"],
                    "arm": arm,
                }
                try:
                    if arm == "direct":
                        outcome = _direct(
                            backend,
                            image,
                            sample["question"],
                            sample["task_type"],
                        )
                    else:
                        outcome = _agent(
                            agent,
                            image,
                            sample["question"],
                            sample["task_type"],
                        )
                    record.update(outcome)
                    record["correct"] = _score(
                        sample["task_type"],
                        record["answer"],
                        sample["reference_answer"],
                    )
                    record["error"] = None
                except Exception as exc:  # noqa: BLE001 - keep paired sweep running
                    record.update(
                        answer="",
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "sample": record["sample_id"],
                            "arm": arm,
                            "answer": record.get("answer"),
                            "reference": record["reference_answer"],
                            "correct": record.get("correct"),
                            "views": [
                                item["view"]
                                for item in record.get("observations", [])
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary: dict[str, Any] = {
        "schema_version": "evors.single-agent-perception-gate.v1",
        "status": "completed",
        "scientific_claim": False,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "expected_model_id": backend.model,
        "seed": seed,
        "offset": offset,
        "grid_sizes": list(grid_sizes),
        "tile_scan_size": tile_scan_size,
        "sample_count": len(samples),
        "results": records,
    }
    for arm in ("direct", "agent"):
        rows = [item for item in records if item["arm"] == arm]
        valid = [item for item in rows if item.get("error") is None]
        summary[f"{arm}_correct"] = sum(bool(item["correct"]) for item in valid)
        summary[f"{arm}_accuracy"] = (
            summary[f"{arm}_correct"] / len(samples) if samples else 0.0
        )
        summary[f"{arm}_errors"] = len(rows) - len(valid)
        summary[f"{arm}_model_calls"] = sum(
            int(item.get("model_calls", 0)) for item in valid
        )
        summary[f"{arm}_tool_calls"] = sum(
            int(item.get("tool_calls", 0)) for item in valid
        )
        for task_type in ("object_quantity", "object_existence"):
            task_rows = [item for item in valid if item["task_type"] == task_type]
            summary[f"{arm}_{task_type}_correct"] = sum(
                bool(item["correct"]) for item in task_rows
            )
            summary[f"{arm}_{task_type}_count"] = len(task_rows)
    summary["gate_passed"] = (
        summary["direct_errors"] == 0
        and summary["agent_errors"] == 0
        and summary["agent_correct"] > summary["direct_correct"]
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--grid-sizes", nargs="*", type=int, default=())
    parser.add_argument("--tile-scan-size", type=int, choices=(2, 3), default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        args.manifest,
        args.model_config,
        args.output_dir,
        offset=args.offset,
        limit=args.limit,
        seed=args.seed,
        grid_sizes=tuple(args.grid_sizes),
        tile_scan_size=args.tile_scan_size,
        overwrite=args.overwrite,
    )
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
