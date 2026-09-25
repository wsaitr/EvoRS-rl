"""Run the integrated EvoRS single-Agent Phase 1-3 episode pipeline."""
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
    LatentFeatureConfig,
    PixelStatsLatentEncoder,
    QwenNativeLatentEncoder,
)
from rs_hmcomm.runtime import (  # noqa: E402
    IntegratedSingleAgentConfig,
    IntegratedSingleAgentRuntime,
)


def _latent_factory(
    mode: str,
) -> Callable[[InMemoryArtifactStore], object] | None:
    if mode == "none":
        return None
    if mode == "pixel_stats":
        return lambda store: PixelStatsLatentEncoder(
            store,
            LatentFeatureConfig(enabled=True),
        )
    if mode == "qwen_native":
        return lambda store: QwenNativeLatentEncoder(store)
    raise ValueError(f"unsupported latent mode: {mode}")


def run(
    manifest: Path,
    model_config: Path,
    output_dir: Path,
    *,
    offset: int = 0,
    limit: int | None = None,
    seed: int = 20260831,
    latent_mode: str = "none",
    evolution_enabled: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    summary_path = output_dir / "summary.json"
    if not overwrite and (events_path.exists() or summary_path.exists()):
        raise FileExistsError("output already exists; pass --overwrite")

    samples = _load_samples(manifest)[offset:]
    if limit is not None:
        samples = samples[:limit]
    backend = _backend(model_config, seed)
    latent_factory = _latent_factory(latent_mode)
    runtime = IntegratedSingleAgentRuntime(
        ActivePerceptionAgent(backend),
        config=IntegratedSingleAgentConfig(
            evolution_enabled=evolution_enabled,
            latent_enabled=latent_factory is not None,
        ),
        latent_encoder_factory=latent_factory,
    )

    records: list[dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record: dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"],
                "question": sample["question"],
                "reference_answer": sample["reference_answer"],
            }
            try:
                with Image.open(sample["image_ref"]) as opened:
                    image = opened.convert("RGB")
                result = runtime.run(
                    image,
                    sample["question"],
                    sample_id=sample["sample_id"],
                    image_ref=str(sample["image_ref"]),
                    task_type=sample["task_type"],
                )
                record.update(result.to_dict(include_private_evolution=True))
                record["correct"] = _score(
                    sample["task_type"],
                    result.answer,
                    sample["reference_answer"],
                )
                record["error"] = None
            except Exception as exc:  # noqa: BLE001 - preserve the sweep
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
                        "error": record.get("error"),
                        "workspace_version": record.get("workspace", {}).get(
                            "version"
                        ),
                        "latent_bytes_loaded": record.get("latent", {}).get(
                            "bytes_loaded", 0
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    valid = [record for record in records if record.get("error") is None]
    passed = len(valid) == len(samples) and all(
        record.get("workspace", {}).get("version") == 2
        and not record.get("workspace_contains_binary_payload", True)
        for record in valid
    )
    summary: dict[str, Any] = {
        "schema_version": "evors.integrated-single-agent-run.v1",
        "status": "passed" if passed else "failed",
        "scientific_claim": False,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "seed": seed,
        "offset": offset,
        "sample_count": len(samples),
        "correct": sum(bool(record.get("correct")) for record in valid),
        "errors": len(samples) - len(valid),
        "latent_mode": latent_mode,
        "learned_visual_representation": latent_mode == "qwen_native",
        "evolution_enabled": evolution_enabled,
        "evolution_episode_count": sum(
            record.get("evolution") is not None for record in valid
        ),
        "model_calls": sum(
            int(record.get("cost", {}).get("model_calls", 0)) for record in valid
        ),
        "tool_calls": sum(
            int(record.get("cost", {}).get("tool_calls", 0)) for record in valid
        ),
        "latent_bytes_loaded": sum(
            int(record.get("latent", {}).get("bytes_loaded", 0))
            for record in valid
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--latent",
        choices=("none", "pixel_stats", "qwen_native"),
        default="none",
    )
    parser.add_argument("--no-evolution", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        args.manifest,
        args.model_config,
        args.output_dir,
        offset=args.offset,
        limit=args.limit,
        seed=args.seed,
        latent_mode=args.latent,
        evolution_enabled=not args.no_evolution,
        overwrite=args.overwrite,
    )
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
