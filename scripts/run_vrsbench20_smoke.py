"""VRSBench-20 real-data smoke: direct QA vs tool-using agent on one local Qwen server.

Two arms share the same backend and samples:
- direct: single whole-image QA call, no tools.
- agent:  Phase 1 SingleAgentRuntime with the crop tool, bounded steps.

Scientific claim: false. This is a pipeline smoke on 20 stratified val samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from rs_hmcomm.agents.generalist import GeneralistAgent
from rs_hmcomm.backends.openai_compatible import OpenAICompatibleVLMBackend
from rs_hmcomm.contracts import BudgetLimits
from rs_hmcomm.runtime.single_agent import SingleAgentRuntime
from rs_hmcomm.tools.crop import CropTool
from rs_hmcomm.tools.registry import ToolRegistry


DEFAULT_MANIFEST = Path("configs/experiments/vrsbench20_manifest.jsonl")
DEFAULT_CONFIG = Path("configs/models/qwen38_27b_q4_local.yaml")
NUM_RE = re.compile(r"^[1-9]$")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_samples(manifest_path: Path) -> list[dict[str, Any]]:
    samples = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            required = {"sample_id", "image_ref", "task_type", "question", "reference_answer"}
            missing = required - set(item)
            if missing:
                raise ValueError(f"{manifest_path}: sample missing fields {sorted(missing)}")
            if item["task_type"] not in ("object_quantity", "object_existence"):
                raise ValueError(f"unsupported task_type: {item['task_type']}")
            samples.append(item)
    if not samples:
        raise ValueError(f"{manifest_path} contains no samples")
    return samples


def score(task_type: str, prediction: str, reference: str) -> bool:
    """Exact-match scoring per task family; normalization is task-local."""
    pred = prediction.strip().lower()
    ref = reference.strip().lower()
    if task_type == "object_quantity":
        match = re.search(r"\d+", pred)
        return bool(match) and match.group(0) == ref
    # object_existence: first yes/no token decides; anything else is wrong.
    match = re.search(r"\b(yes|no)\b", pred)
    return bool(match) and match.group(1) == ref


def extract_answer(text: str) -> str:
    return text.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_backend(config_path: Path) -> OpenAICompatibleVLMBackend:
    values = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    server = _mapping(values.get("server"), "server")
    identity = _mapping(values.get("identity"), "identity")
    return OpenAICompatibleVLMBackend(
        identity["api_alias"],
        base_url=f"http://{server['host']}:{server['port']}/v1",
        api_key="local",
        timeout=180,
        max_tokens=256,
        enable_thinking=False,
        preserve_thinking=False,
    )


def run_direct(
    backend: OpenAICompatibleVLMBackend,
    image: Image.Image,
    question: str,
) -> dict[str, Any]:
    from rs_hmcomm.backends.base import BackendRequest, VisualInput

    start = time.perf_counter()
    request = BackendRequest(
        prompt=(
            "You are a remote-sensing image analysis assistant. "
            "Answer the question about the image concisely. "
            "For counting questions answer with a single number. "
            "For yes/no questions answer with exactly Yes or No.\n\n"
            f"Question: {question}\nAnswer:"
        ),
        visuals=(VisualInput(visual_id="image", label="whole_image", image=image),),
    )
    output = backend.generate_turn(request)
    wall_ms = (time.perf_counter() - start) * 1000.0
    usage = output.usage
    return {
        "answer": extract_answer(output.text),
        "model_calls": 1,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "tool_calls": 0,
        "wall_time_ms": wall_ms,
    }


def run_agent_arm(
    backend: OpenAICompatibleVLMBackend,
    image: Image.Image,
    question: str,
    *,
    max_steps: int = 4,
) -> dict[str, Any]:
    registry = ToolRegistry((CropTool(),))
    budget = BudgetLimits(
        max_agent_calls=max_steps + 2,
        max_tool_calls=3,
        max_model_calls=max_steps + 2,
        max_model_input_tokens=200_000,
        max_model_output_tokens=4_096,
        max_billed_compute_ms=600_000.0,
        max_wall_time_ms=600_000.0,
        max_errors=2,
        max_retries=1,
        max_invalid_actions=2,
    )
    runtime = SingleAgentRuntime(
        GeneralistAgent(backend),
        registry,
        budget,
        max_steps=max_steps,
    )
    result = runtime.run(
        image,
        question,
        require_tool=True,
        answer_format="short_text",
    )
    usage = result.cost
    return {
        "answer": extract_answer(result.answer or ""),
        "model_calls": usage.model_calls,
        "input_tokens": usage.model_input_tokens,
        "output_tokens": usage.model_output_tokens,
        "tool_calls": usage.tool_calls,
        "wall_time_ms": result.wall_time_ms,
        "stop_reason": result.stop_reason.value,
        "agent_events": len(result.trace.events),
    }


DIRECT_PROMPT_TOKEN_ESTIMATE = 60  # reported separately from measured server usage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arms", nargs="+", choices=("direct", "agent"), default=("direct", "agent"))
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]

    backend = build_backend(args.model_config)

    events_path = output_dir / "events.jsonl"
    results: list[dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8") as event_log:
        for sample in samples:
            image_path = Path(sample["image_ref"])
            if not image_path.exists():
                raise FileNotFoundError(f"missing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            for arm in args.arms:
                record = {
                    "sample_id": sample["sample_id"],
                    "annotation": sample.get("annotation"),
                    "task_type": sample["task_type"],
                    "question": sample["question"],
                    "reference_answer": sample["reference_answer"],
                    "arm": arm,
                    "image_size": list(image.size),
                }
                try:
                    if arm == "direct":
                        outcome = run_direct(backend, image, sample["question"])
                    else:
                        outcome = run_agent_arm(backend, image, sample["question"])
                    record.update(outcome)
                    record["correct"] = score(
                        sample["task_type"], record["answer"], sample["reference_answer"]
                    )
                    record["error"] = None
                except Exception as exc:  # noqa: BLE001 - log and continue the sweep
                    record.update(
                        {
                            "answer": None,
                            "correct": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                results.append(record)
                event_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                event_log.flush()
                print(
                    json.dumps(
                        {
                            "sample_id": record["sample_id"],
                            "arm": arm,
                            "correct": record.get("correct"),
                            "answer": record.get("answer"),
                            "ref": record["reference_answer"],
                            "tokens_in": record.get("input_tokens"),
                            "tokens_out": record.get("output_tokens"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary: dict[str, Any] = {
        "schema_version": "evors.vrsbench20-smoke-summary.v1",
        "phase": "post-phase4-realdata-smoke",
        "status": "completed",
        "scientific_claim": False,
        "validation_scope": "real_data_pipeline_smoke",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "model_config": str(args.model_config),
        "backend_model": backend.model,
        "base_url": backend.endpoint,
        "sample_count": len(samples),
        "arms": list(args.arms),
        "results": results,
    }
    for arm in args.arms:
        arm_rows = [r for r in results if r["arm"] == arm and r.get("error") is None]
        if arm_rows:
            summary[f"{arm}_accuracy"] = sum(r["correct"] for r in arm_rows) / len(arm_rows)
            summary[f"{arm}_mean_input_tokens"] = sum(
                r.get("input_tokens") or 0 for r in arm_rows
            ) / len(arm_rows)
            summary[f"{arm}_mean_output_tokens"] = sum(
                r.get("output_tokens") or 0 for r in arm_rows
            ) / len(arm_rows)
            summary[f"{arm}_mean_wall_time_ms"] = sum(
                r.get("wall_time_ms") or 0.0 for r in arm_rows
            ) / len(arm_rows)
    errors = [r for r in results if r.get("error")]
    summary["error_count"] = len(errors)

    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
