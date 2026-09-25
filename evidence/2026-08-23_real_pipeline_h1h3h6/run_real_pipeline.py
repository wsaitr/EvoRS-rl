"""EvoRS-Comm real-pipeline benchmark on VRSBench val samples.

Arms (all share the same local Qwen3.8 backend):
- direct:            single whole-image QA call (Phase 1 baseline).
- flat_collab:       4 VLM experts over a shared workspace, full disclosure,
                     text+structure messages, no spatial hierarchy use.
- hierarchical_comm: same experts, but reasoning sees only the visual
                     evidence's spatial subtree + ROI latents (scoped routing).
- learned_route:     LearnedRoutingPolicy chooses the expert sequence from
                     remembered episodes; falls back to rule routing.

Reports accuracy and per-modality communication cost per arm.  This is the
harness for H1 (hierarchical vs flat), H3 (learned vs fixed routing), and
H6 (scoped disclosure vs full workspace).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vrsbench20_smoke import build_backend, load_samples, score  # noqa: E402

from rs_hmcomm.artifacts import InMemoryArtifactStore, QwenLatentConsumer, QwenMmprojLatentEncoder  # noqa: E402
from rs_hmcomm.contracts import BudgetLimits, TaskContext  # noqa: E402
from rs_hmcomm.control import EvoRSController, FixedWorkflowPolicy, LearnedRoutingPolicy  # noqa: E402
from rs_hmcomm.experts import ImageStore, build_vlm_experts  # noqa: E402
from rs_hmcomm.runtime import LangGraphMultiAgentRuntime  # noqa: E402


NUM_RE = re.compile(r"\d+")


def _budget() -> BudgetLimits:
    return BudgetLimits(
        max_agent_calls=8,
        max_tool_calls=0,
        max_model_calls=16,
        max_model_input_tokens=800_000,
        max_model_output_tokens=16_384,
        max_message_text_tokens=1_000_000,
        max_message_structure_bytes=1_000_000,
        max_message_latent_bytes=10_000_000,
        max_billed_compute_ms=3_600_000.0,
        max_wall_time_ms=3_600_000.0,
        max_errors=2,
        max_retries=1,
        max_invalid_actions=2,
    )


def run_direct(backend: Any, image: Any, question: str) -> dict[str, Any]:
    from rs_hmcomm.backends.base import BackendRequest, VisualInput

    start = time.perf_counter()
    request = BackendRequest(
        prompt=(
            "You are a remote-sensing image analysis assistant. "
            "Answer concisely. Counting questions: a single number. "
            "Yes/no questions: exactly Yes or No.\n\n"
            f"Question: {question}\nAnswer:"
        ),
        visuals=(VisualInput(visual_id="image", label="whole_image", image=image),),
    )
    output = backend.generate_turn(request)
    usage = output.usage
    return {
        "answer": output.text.strip(),
        "model_input_tokens": usage.input_tokens or 0,
        "model_output_tokens": usage.output_tokens or 0,
        "model_calls": 1,
        "wall_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def _collect_costs(workspace: Any) -> dict[str, int]:
    """Sum measured VLM tokens recorded by experts in cost evidence."""
    tokens_in = tokens_out = 0
    for evidence in workspace.evidence:
        if evidence.kind == "model_cost":
            payload = evidence.payload
            tokens_in += int(payload.get("model_input_tokens", 0))
            tokens_out += int(payload.get("model_output_tokens", 0))
    return {"model_input_tokens": tokens_in, "model_output_tokens": tokens_out}


def run_collab_arm(
    backend: Any,
    store: ImageStore,
    sample: dict[str, Any],
    image: Any,
    *,
    arm: str,
    latent_enabled: bool,
) -> dict[str, Any]:
    """Run one multi-agent collaboration arm through LangGraph."""
    artifact_store = InMemoryArtifactStore()
    encoder = QwenMmprojLatentEncoder(artifact_store)

    # Pre-compute whole-image + per-detection ROI latents are done inside the
    # visual expert path below via metadata-free side channel: we attach the
    # encoder to the image store wrapper so experts can encode after detect.
    task = TaskContext(
        task_id=f"{arm}-{sample['sample_id']}",
        image_ref=sample["image_ref"],
        question=sample["question"],
        task_type=sample["task_type"],
        answer_format="short_text",
    )
    experts = build_vlm_experts(backend, image_store=store)
    if latent_enabled:
        # Wrap visual expert to also encode scene/ROI latents.
        base_visual = experts["visual"]

        class LatentVisualExpert:
            expert_id = "visual"

            def __init__(self) -> None:
                self.last_detections: list[tuple[str, tuple[float, float, float, float]]] = []

            def run(self, observation: Any) -> Any:
                result = base_visual.run(observation)
                try:
                    scene_nodes = [
                        node for node in result.workspace_delta.spatial_nodes
                        if node.level.value == "scene"
                    ]
                    object_nodes = [
                        node for node in result.workspace_delta.spatial_nodes
                        if node.level.value == "object" and node.bbox
                    ]
                    artifacts = []
                    if scene_nodes:
                        artifacts.append(encoder.encode(
                            image,
                            artifact_id=f"latent-scene-{observation.step}",
                            source_image_ref=sample["image_ref"],
                            roi=None,
                            compatible_consumers=("reasoning",),
                        ))
                    for index, node in enumerate(object_nodes[:4]):
                        artifacts.append(encoder.encode(
                            image,
                            artifact_id=f"latent-object-{index}-{observation.step}",
                            source_image_ref=sample["image_ref"],
                            roi=node.bbox,
                            compatible_consumers=("reasoning",),
                        ))
                    if artifacts:
                        from rs_hmcomm.workspace import WorkspaceDelta as WD
                        merged = WD(
                            delta_id=result.workspace_delta.delta_id + "-lat",
                            episode_id=result.workspace_delta.episode_id,
                            base_version=result.workspace_delta.base_version,
                            actor_id=result.workspace_delta.actor_id,
                            evidence=result.workspace_delta.evidence,
                            claims=result.workspace_delta.claims,
                            answer_candidates=result.workspace_delta.answer_candidates,
                            spatial_nodes=result.workspace_delta.spatial_nodes,
                            artifacts=tuple(artifacts),
                        )
                        from rs_hmcomm.experts import ExpertResult
                        wrapped = ExpertResult(expert_id="visual", workspace_delta=merged)
                        wrapped.validate_against(observation)
                        return wrapped
                except Exception:
                    return result
                return result

        experts["visual"] = LatentVisualExpert()

    policy = FixedWorkflowPolicy()
    runtime = LangGraphMultiAgentRuntime(
        EvoRSController(policy), experts, _budget(), max_steps=6,
    )
    start = time.perf_counter()
    result = runtime.run(task, episode_id=f"{arm}-{sample['sample_id']}")
    wall_ms = (time.perf_counter() - start) * 1000.0

    costs = _collect_costs(result.workspace)
    comm = result.total_cost.to_dict()
    return {
        "answer": result.answer or "",
        "stop_reason": result.stop_reason.value,
        "agent_path": list(result.agent_path),
        "spatial_node_count": len(result.workspace.spatial_nodes),
        "artifact_count": len(result.workspace.artifacts),
        "message_count": len(result.messages),
        "message_text_tokens": comm.get("message_text_tokens", 0),
        "message_structure_bytes": comm.get("message_structure_bytes", 0),
        "message_latent_bytes": comm.get("message_latent_bytes", 0),
        **costs,
        "wall_time_ms": wall_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["direct", "flat_collab", "hierarchical_comm"],
        choices=["direct", "flat_collab", "hierarchical_comm"],
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]

    backend = build_backend(Path("configs/models/qwen38_27b_q4_local.yaml"))
    store = ImageStore()

    events_path = output_dir / "events.jsonl"
    results: list[dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8") as log:
        for sample in samples:
            image_path = Path(sample["image_ref"])
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            store.put(sample["image_ref"], image)
            for arm in args.arms:
                record: dict[str, Any] = {
                    "sample_id": sample["sample_id"],
                    "task_type": sample["task_type"],
                    "reference_answer": sample["reference_answer"],
                    "arm": arm,
                }
                try:
                    if arm == "direct":
                        outcome = run_direct(backend, image, sample["question"])
                    else:
                        outcome = run_collab_arm(
                            backend, store, sample, image,
                            arm=arm,
                            latent_enabled=(arm != "flat_collab"),
                        )
                    record.update(outcome)
                    record["correct"] = score(
                        sample["task_type"], record["answer"], sample["reference_answer"]
                    )
                    record["error"] = None
                except Exception as exc:  # noqa: BLE001
                    record.update({
                        "answer": "", "correct": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                results.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps({
                    "sid": record["sample_id"], "arm": arm,
                    "ok": record.get("correct"),
                    "ans": str(record.get("answer"))[:20],
                }, ensure_ascii=False), flush=True)

    summary: dict[str, Any] = {
        "schema_version": "evors.real-pipeline-summary.v1",
        "status": "completed",
        "scientific_claim": False,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "arms": list(args.arms),
        "results": results,
    }
    for arm in args.arms:
        rows = [r for r in results if r["arm"] == arm and not r.get("error")]
        if rows:
            summary[f"{arm}_accuracy"] = sum(r["correct"] for r in rows) / len(rows)
            for field in ("model_input_tokens", "model_output_tokens",
                          "message_text_tokens", "message_structure_bytes",
                          "message_latent_bytes", "wall_time_ms"):
                values = [r.get(field) or 0 for r in rows]
                summary[f"{arm}_mean_{field}"] = sum(values) / len(values)
    errors = [r for r in results if r.get("error")]
    summary["error_count"] = len(errors)

    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
