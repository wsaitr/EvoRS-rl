"""Run real Qwen multi-Agent inference with Phase 4 rule or Phase 5 learned routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_complete_single_agent import _latent_factory  # noqa: E402
from scripts.run_single_agent_perception_gate import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_CONFIG,
    _backend,
    _load_samples,
    _score,
    _sha256,
)
from rs_hmcomm.agents import ActivePerceptionAgent  # noqa: E402
from rs_hmcomm.artifacts import (  # noqa: E402
    InMemoryArtifactStore,
    QwenLatentConsumer,
)
from rs_hmcomm.contracts import BudgetLimits, TaskContext  # noqa: E402
from rs_hmcomm.control import EvolutionController, RuleRoutingPolicy  # noqa: E402
from rs_hmcomm.experts import (  # noqa: E402
    EvolvingVisualExpert,
    ImageStore,
    build_vlm_experts,
)
from rs_hmcomm.runtime import (  # noqa: E402
    IntegratedSingleAgentConfig,
    IntegratedSingleAgentRuntime,
    LangGraphMultiAgentRuntime,
)


def _budget() -> BudgetLimits:
    return BudgetLimits(
        max_agent_calls=8,
        max_tool_calls=64,
        max_model_calls=64,
        max_model_input_tokens=1_000_000,
        max_model_output_tokens=32_768,
        max_message_text_tokens=1_000_000,
        max_message_structure_bytes=2_000_000,
        max_message_latent_bytes=32_000_000,
        max_billed_compute_ms=3_600_000.0,
        max_wall_time_ms=3_600_000.0,
        max_errors=2,
        max_retries=1,
        max_invalid_actions=2,
    )


def _build_experts(
    backend: Any,
    image_store: ImageStore,
    artifact_store: InMemoryArtifactStore,
    *,
    latent_mode: str,
) -> dict[str, Any]:
    latent_factory: Callable[[InMemoryArtifactStore], object] | None = _latent_factory(
        latent_mode
    )
    experts = build_vlm_experts(
        backend,
        image_store=image_store,
        artifact_store=artifact_store if latent_factory is not None else None,
        latent_consumer_factory=(
            QwenLatentConsumer if latent_factory is not None else None
        ),
        include_detection_context=True,
    )
    inner = IntegratedSingleAgentRuntime(
        ActivePerceptionAgent(backend),
        config=IntegratedSingleAgentConfig(
            evolution_enabled=True,
            latent_enabled=latent_factory is not None,
            latent_consumer_id="reasoning",
        ),
        latent_encoder_factory=latent_factory,
    )
    experts["visual"] = EvolvingVisualExpert(
        inner,
        image_store=image_store,
        artifact_store=artifact_store,
    )
    return experts


def run(
    manifest: Path,
    model_config: Path,
    output_dir: Path,
    *,
    offset: int = 0,
    limit: int | None = 1,
    seed: int = 20260901,
    latent_mode: str = "none",
    routing_checkpoint: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    summary_path = output_dir / "summary.json"
    if not overwrite and (events_path.exists() or summary_path.exists()):
        raise FileExistsError("output already exists; pass --overwrite")
    samples = _load_samples(manifest)[offset:]
    if limit is not None:
        samples = samples[:limit]
    unsupported = [
        item["task_type"]
        for item in samples
        if item["task_type"] not in {"object_quantity", "object_existence"}
    ]
    if unsupported:
        raise ValueError(
            "evolving Visual Expert currently supports quantity/existence tasks; "
            f"got {unsupported}"
        )

    backend = _backend(model_config, seed)
    image_store = ImageStore()
    records: list[dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            with Image.open(sample["image_ref"]) as opened:
                image = opened.convert("RGB")
            image_store.put(str(sample["image_ref"]), image)
            artifact_store = InMemoryArtifactStore()
            if routing_checkpoint is None:
                routing_policy = RuleRoutingPolicy()
            else:
                from rs_hmcomm.control import LearnedRoutingPolicy

                routing_policy = LearnedRoutingPolicy.from_checkpoint(
                    routing_checkpoint,
                    seed=seed,
                )
                routing_policy.begin_episode(training=False, explore=False)
            runtime = LangGraphMultiAgentRuntime(
                EvolutionController(routing_policy),
                _build_experts(
                    backend,
                    image_store,
                    artifact_store,
                    latent_mode=latent_mode,
                ),
                _budget(),
                max_steps=8,
                disclosure_mode="scoped",
            )
            task = TaskContext(
                task_id=f"phase4-{sample['sample_id']}",
                image_ref=str(sample["image_ref"]),
                question=sample["question"],
                task_type=sample["task_type"],
                answer_format=(
                    "integer"
                    if sample["task_type"] == "object_quantity"
                    else "yes_no"
                ),
                # No reference answer, mock label, or GT bbox enters the task.
                metadata={"dataset_sample_id": sample["sample_id"]},
            )
            record: dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "reference_answer": sample["reference_answer"],
            }
            try:
                result = runtime.run(task, episode_id=f"phase4-{sample['sample_id']}")
                record.update(result.to_dict())
                record["correct"] = _score(
                    sample["task_type"],
                    result.answer or "",
                    sample["reference_answer"],
                )
                record["artifact_store_size"] = len(artifact_store)
                record["error"] = None
            except Exception as exc:  # noqa: BLE001 - persist experimental failure
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
                        "answer": record.get("answer"),
                        "reference": record["reference_answer"],
                        "correct": record.get("correct"),
                        "agent_path": record.get("agent_path"),
                        "error": record.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    valid = [item for item in records if item.get("error") is None]
    passed = len(valid) == len(samples) and all(
        item.get("stop_reason") == "completed" and item.get("correct") is True
        for item in valid
    )
    summary = {
        "schema_version": "evors.real-multi-agent.v1",
        "status": "passed" if passed else "failed",
        "scientific_claim": False,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "seed": seed,
        "offset": offset,
        "sample_count": len(samples),
        "correct": sum(bool(item.get("correct")) for item in valid),
        "errors": len(samples) - len(valid),
        "latent_mode": latent_mode,
        "controller": "EvolutionController",
        "routing_policy": (
            "RuleRoutingPolicy"
            if routing_checkpoint is None
            else "LearnedRoutingPolicy"
        ),
        "routing_checkpoint": (
            None if routing_checkpoint is None else str(routing_checkpoint)
        ),
        "routing_checkpoint_sha256": (
            None if routing_checkpoint is None else _sha256(routing_checkpoint)
        ),
        "results": records,
    }
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
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--latent",
        choices=("none", "pixel_stats", "qwen_native"),
        default="none",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--routing-checkpoint",
        type=Path,
        help="Phase 5 LearnedRoutingPolicy checkpoint; omit for RuleRoutingPolicy",
    )
    args = parser.parse_args(argv)
    summary = run(
        args.manifest,
        args.model_config,
        args.output_dir,
        offset=args.offset,
        limit=args.limit,
        seed=args.seed,
        latent_mode=args.latent,
        routing_checkpoint=args.routing_checkpoint,
        overwrite=args.overwrite,
    )
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
