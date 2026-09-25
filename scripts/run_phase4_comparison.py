"""Run a resumable paired Direct/Single/Multi EvoRS comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import re
import math
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_real_multi_agent import _budget  # noqa: E402
from scripts.run_single_agent_perception_gate import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_CONFIG,
    _backend,
    _load_samples,
    _score,
    _sha256,
)
from rs_hmcomm.agents import (  # noqa: E402
    ActivePerceptionAgent,
    ActivePerceptionResult,
    PerceptionCall,
)
from rs_hmcomm.artifacts import InMemoryArtifactStore  # noqa: E402
from rs_hmcomm.contracts import TaskContext  # noqa: E402
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


DEFAULT_BASELINE_EVENTS = Path(
    "evidence/2026-09-01_single_agent_perception_gate/events.jsonl"
)
ARM_NAMES = ("direct_vlm", "single_agent", "multi_agent")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(item)
    return rows


def _baseline_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _jsonl(path):
        raw_arm = item.get("arm")
        arm = "single_agent" if raw_arm == "agent" else "direct_vlm"
        if arm not in {"direct_vlm", "single_agent"}:
            continue
        key = (str(item.get("sample_id", "")), arm)
        if not key[0] or key in index:
            raise ValueError(f"invalid or duplicate baseline record: {key}")
        if item.get("error") is not None:
            raise ValueError(f"baseline record contains an error: {key}")
        index[key] = item
    return index


class _NoBackend:
    def generate_turn(self, request):  # pragma: no cover - defensive
        raise AssertionError("recorded perception must not call a backend")


class RecordedPerceptionAgent(ActivePerceptionAgent):
    """Replay a previously measured real VLM perception trace exactly once."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(_NoBackend())
        if record.get("arm") != "agent" or record.get("error") is not None:
            raise ValueError("record must be a successful active-perception baseline")
        calls = tuple(
            PerceptionCall(
                view=item["view"],
                answer=item["answer"],
                input_tokens=int(item["input_tokens"]),
                output_tokens=int(item["output_tokens"]),
                latency_ms=float(item["latency_ms"]),
                model_id=item["model_id"],
            )
            for item in record["observations"]
        )
        original_boxes: list[tuple[float, float, float, float]] = []
        enhanced_boxes: list[tuple[float, float, float, float]] = []
        tiled_boxes: list[tuple[float, float, float, float]] = []
        for call in calls:
            parsed = self._localized_boxes(call.answer) or []
            if call.view == "object_localization":
                original_boxes.extend(parsed)
            elif call.view == "enhanced_object_localization":
                enhanced_boxes.extend(parsed)
            elif call.view.startswith("tile_localization_"):
                match = re.search(r"tile_r(\d+)c(\d+)", call.view)
                if match is not None:
                    row, column = (int(value) for value in match.groups())
                    tile_bbox = (
                        column / 2.0,
                        row / 2.0,
                        (column + 1) / 2.0,
                        (row + 1) / 2.0,
                    )
                    tiled_boxes.extend(
                        self._map_local_box(box, tile_bbox) for box in parsed
                    )
        decision_rule = str(record["decision_rule"])
        if decision_rule == "global_scale_consensus":
            boxes = tuple(original_boxes)
        elif decision_rule == "enhanced_count_localization_consensus":
            boxes = tuple(enhanced_boxes)
        else:
            boxes = tuple(
                self._deduplicate_boxes(original_boxes + enhanced_boxes + tiled_boxes)
            )
        self.record = record
        self.result = ActivePerceptionResult(
            answer=str(record["answer"]),
            calls=calls,
            tool_calls=int(record["tool_calls"]),
            decision_rule=decision_rule,
            wall_time_ms=float(record["wall_time_ms"]),
            candidate_boxes=boxes,
        )
        self._used = False

    def run(self, image, question, task_type):
        if self._used:
            raise RuntimeError("RecordedPerceptionAgent may be consumed only once")
        if question != self.record["question"] or task_type != self.record["task_type"]:
            raise ValueError("replay task does not match the baseline record")
        self._used = True
        return self.result


def _cost(
    *,
    model_calls: int,
    tool_calls: int,
    input_tokens: int,
    output_tokens: int,
    wall_time_ms: float,
    agent_calls: int,
    message_text_tokens: int = 0,
    message_structure_bytes: int = 0,
    message_latent_bytes: int = 0,
) -> dict[str, int | float]:
    return {
        "agent_calls": agent_calls,
        "tool_calls": tool_calls,
        "model_calls": model_calls,
        "model_input_tokens": input_tokens,
        "model_output_tokens": output_tokens,
        "wall_time_ms": wall_time_ms,
        "message_text_tokens": message_text_tokens,
        "message_structure_bytes": message_structure_bytes,
        "message_latent_bytes": message_latent_bytes,
    }


def _direct_arm(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": str(record["answer"]),
        "correct": bool(record["correct"]),
        "error": None,
        "cost": _cost(
            agent_calls=0,
            model_calls=int(record["model_calls"]),
            tool_calls=int(record["tool_calls"]),
            input_tokens=int(record["input_tokens"]),
            output_tokens=int(record["output_tokens"]),
            wall_time_ms=float(record["wall_time_ms"]),
        ),
        "source_event": record,
    }


def _single_arm(
    sample: dict[str, Any],
    image: Image.Image,
    perception_record: dict[str, Any],
) -> dict[str, Any]:
    replay = RecordedPerceptionAgent(perception_record)
    runtime = IntegratedSingleAgentRuntime(
        replay,
        config=IntegratedSingleAgentConfig(evolution_enabled=True),
    )
    started = time.perf_counter()
    result = runtime.run(
        image,
        sample["question"],
        sample_id=sample["sample_id"],
        image_ref=str(sample["image_ref"]),
        task_type=sample["task_type"],
    )
    reconstruction_ms = (time.perf_counter() - started) * 1000.0
    serialized = result.to_dict(include_private_evolution=True)
    return {
        "answer": result.answer,
        "correct": _score(
            sample["task_type"], result.answer, sample["reference_answer"]
        ),
        "error": None,
        "cost": _cost(
            agent_calls=1,
            model_calls=result.cost.model_calls,
            tool_calls=result.cost.tool_calls,
            input_tokens=result.cost.model_input_tokens,
            output_tokens=result.cost.model_output_tokens,
            wall_time_ms=float(perception_record["wall_time_ms"]) + reconstruction_ms,
        ),
        "reconstruction_wall_time_ms": reconstruction_ms,
        "result": serialized,
    }


def _multi_arm(
    sample: dict[str, Any],
    image: Image.Image,
    perception_record: dict[str, Any],
    backend: Any,
) -> dict[str, Any]:
    image_store = ImageStore()
    image_store.put(str(sample["image_ref"]), image)
    artifact_store = InMemoryArtifactStore()
    experts = build_vlm_experts(backend, image_store=image_store)
    visual_runtime = IntegratedSingleAgentRuntime(
        RecordedPerceptionAgent(perception_record),
        config=IntegratedSingleAgentConfig(evolution_enabled=True),
    )
    experts["visual"] = EvolvingVisualExpert(
        visual_runtime,
        image_store=image_store,
        artifact_store=artifact_store,
    )
    runtime = LangGraphMultiAgentRuntime(
        EvolutionController(RuleRoutingPolicy()),
        experts,
        _budget(),
        max_steps=8,
        disclosure_mode="scoped",
    )
    task = TaskContext(
        task_id=f"phase4-compare-{sample['sample_id']}",
        image_ref=str(sample["image_ref"]),
        question=sample["question"],
        task_type=sample["task_type"],
        answer_format=(
            "integer" if sample["task_type"] == "object_quantity" else "yes_no"
        ),
        metadata={"dataset_sample_id": sample["sample_id"]},
    )
    started = time.perf_counter()
    result = runtime.run(task, episode_id=f"phase4-compare-{sample['sample_id']}")
    incremental_ms = (time.perf_counter() - started) * 1000.0
    usage = result.total_cost
    return {
        "answer": result.answer or "",
        "correct": _score(
            sample["task_type"], result.answer or "", sample["reference_answer"]
        ),
        "error": None,
        "cost": _cost(
            agent_calls=usage.agent_calls,
            model_calls=usage.model_calls,
            tool_calls=usage.tool_calls,
            input_tokens=usage.model_input_tokens,
            output_tokens=usage.model_output_tokens,
            wall_time_ms=float(perception_record["wall_time_ms"]) + incremental_ms,
            message_text_tokens=usage.message_text_tokens,
            message_structure_bytes=usage.message_structure_bytes,
            message_latent_bytes=usage.message_latent_bytes,
        ),
        "incremental_collaboration_wall_time_ms": incremental_ms,
        "artifact_store_size": len(artifact_store),
        "result": result.to_dict(),
    }


def _safe_arm(callable_) -> dict[str, Any]:
    try:
        return callable_()
    except Exception as exc:  # noqa: BLE001 - experimental evidence must persist
        return {
            "answer": "",
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "cost": _cost(
                agent_calls=0,
                model_calls=0,
                tool_calls=0,
                input_tokens=0,
                output_tokens=0,
                wall_time_ms=0.0,
            ),
        }


def _arm_stats(records: Iterable[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [item["arms"][arm] for item in records]
    valid = [item for item in rows if item.get("error") is None]
    costs = [item["cost"] for item in valid]
    metrics = tuple(costs[0]) if costs else tuple(_cost(
        agent_calls=0, model_calls=0, tool_calls=0, input_tokens=0,
        output_tokens=0, wall_time_ms=0.0,
    ))
    total = {
        key: sum(float(item[key]) for item in costs)
        for key in metrics
    }
    correct = sum(bool(item["correct"]) for item in valid)
    confidence_interval = _wilson_interval(correct, len(valid)) if valid else None
    return {
        "completed": len(valid),
        "errors": len(rows) - len(valid),
        "correct": correct,
        "accuracy": (correct / len(valid) if valid else None),
        "accuracy_wilson_95": confidence_interval,
        "total_cost": total,
        "mean_cost": {
            key: (value / len(valid) if valid else None)
            for key, value in total.items()
        },
    }


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def _paired(records: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    valid = [
        item
        for item in records
        if item["arms"][left].get("error") is None
        and item["arms"][right].get("error") is None
    ]
    wins = [
        item["sample_id"]
        for item in valid
        if item["arms"][left]["correct"] and not item["arms"][right]["correct"]
    ]
    losses = [
        item["sample_id"]
        for item in valid
        if not item["arms"][left]["correct"] and item["arms"][right]["correct"]
    ]
    changed = [
        {
            "sample_id": item["sample_id"],
            "left_answer": item["arms"][left]["answer"],
            "right_answer": item["arms"][right]["answer"],
            "reference_answer": item["reference_answer"],
        }
        for item in valid
        if str(item["arms"][left]["answer"]).casefold()
        != str(item["arms"][right]["answer"]).casefold()
    ]
    return {
        "left": left,
        "right": right,
        "paired_samples": len(valid),
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(valid) - len(wins) - len(losses),
        "exact_mcnemar_p_two_sided": _exact_mcnemar_p(len(wins), len(losses)),
        "win_sample_ids": wins,
        "loss_sample_ids": losses,
        "changed_answers": changed,
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    expected_count: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    arms = {arm: _arm_stats(records, arm) for arm in ARM_NAMES}
    by_task_type: dict[str, dict[str, Any]] = {}
    for task_type in sorted({item["task_type"] for item in records}):
        subset = [item for item in records if item["task_type"] == task_type]
        by_task_type[task_type] = {
            arm: _arm_stats(subset, arm) for arm in ARM_NAMES
        }
    complete = len(records) == expected_count and all(
        item["arms"][arm].get("error") is None
        for item in records
        for arm in ARM_NAMES
    )
    expected_path = ["visual", "reasoning", "verification"]
    integrity = {
        "unique_sample_ids": len({item["sample_id"] for item in records}) == len(records),
        "all_arms_error_free": all(
            item["arms"][arm].get("error") is None
            for item in records
            for arm in ARM_NAMES
        ),
        "multi_agent_paths_valid": all(
            item["arms"]["multi_agent"].get("error") is not None
            or item["arms"]["multi_agent"]["result"]["agent_path"] == expected_path
            for item in records
        ),
        "multi_agent_task_label_isolation": all(
            "reference_answer"
            not in item["arms"]["multi_agent"]["result"]["task"]["metadata"]
            for item in records
            if item["arms"]["multi_agent"].get("error") is None
        ),
        "multi_agent_replay_material_present": all(
            bool(item["arms"]["multi_agent"]["result"]["transitions"])
            and bool(item["arms"]["multi_agent"]["result"]["messages"])
            for item in records
            if item["arms"]["multi_agent"].get("error") is None
        ),
    }
    return {
        "schema_version": "evors.phase4-paired-comparison.v1",
        "status": "completed" if complete else "running",
        "scientific_claim": False,
        "design": "paired_frozen_perception",
        "expected_sample_count": expected_count,
        "completed_sample_count": len(records),
        "provenance": provenance,
        "arms": arms,
        "by_task_type": by_task_type,
        "integrity": integrity,
        "paired": {
            "single_vs_direct": _paired(records, "single_agent", "direct_vlm"),
            "multi_vs_direct": _paired(records, "multi_agent", "direct_vlm"),
            "multi_vs_single": _paired(records, "multi_agent", "single_agent"),
        },
    }


def _write_csvs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "sample_id", "task_type", "question", "reference_answer", "arm",
            "answer", "correct", "error", "agent_calls", "tool_calls",
            "model_calls", "model_input_tokens", "model_output_tokens",
            "wall_time_ms", "message_text_tokens", "message_structure_bytes",
            "message_latent_bytes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            for arm in ARM_NAMES:
                arm_result = item["arms"][arm]
                writer.writerow({
                    "sample_id": item["sample_id"],
                    "task_type": item["task_type"],
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "arm": arm,
                    "answer": arm_result["answer"],
                    "correct": arm_result["correct"],
                    "error": arm_result.get("error"),
                    **arm_result["cost"],
                })
    with (output_dir / "accuracy_cost.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "arm", "completed", "correct", "accuracy", "accuracy_ci95_lower",
            "accuracy_ci95_upper", *next(iter(summary["arms"].values()))["mean_cost"]
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ARM_NAMES:
            stats = summary["arms"][arm]
            writer.writerow({
                "arm": arm,
                "completed": stats["completed"],
                "correct": stats["correct"],
                "accuracy": stats["accuracy"],
                "accuracy_ci95_lower": stats["accuracy_wilson_95"][0],
                "accuracy_ci95_upper": stats["accuracy_wilson_95"][1],
                **stats["mean_cost"],
            })


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# EvoRS Phase 4 paired comparison",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Design: paired frozen perception. Reference answers were used only by the scorer.",
        "",
        "## Accuracy and mean cost",
        "",
        "| Arm | Correct | Accuracy (Wilson 95% CI) | Model calls | Tool calls | Wall time (ms) | Text tokens | Structure bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        stats = summary["arms"][arm]
        cost = stats["mean_cost"]
        accuracy = "n/a" if stats["accuracy"] is None else f"{stats['accuracy']:.3f}"
        interval = stats["accuracy_wilson_95"]
        ci = "n/a" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        lines.append(
            f"| {arm} | {stats['correct']}/{stats['completed']} | {accuracy} {ci} | "
            f"{cost['model_calls'] or 0:.2f} | {cost['tool_calls'] or 0:.2f} | "
            f"{cost['wall_time_ms'] or 0:.1f} | {cost['message_text_tokens'] or 0:.1f} | "
            f"{cost['message_structure_bytes'] or 0:.1f} |"
        )
    lines.extend(["", "## Paired outcomes", ""])
    for name, item in summary["paired"].items():
        lines.append(
            f"- `{name}`: wins {item['wins']}, losses {item['losses']}, "
            f"ties {item['ties']} over {item['paired_samples']} samples; exact "
            f"McNemar p={item['exact_mcnemar_p_two_sided']:.3f}."
        )
    lines.extend([
        "",
        "## Task-type accuracy",
        "",
        "| Task type | Direct | Single | Multi |",
        "|---|---:|---:|---:|",
    ])
    for task_type, arms in summary["by_task_type"].items():
        lines.append(
            f"| {task_type} | {arms['direct_vlm']['correct']}/{arms['direct_vlm']['completed']} | "
            f"{arms['single_agent']['correct']}/{arms['single_agent']['completed']} | "
            f"{arms['multi_agent']['correct']}/{arms['multi_agent']['completed']} |"
        )
    lines.extend(["", "## Changed Multi-Agent answers", ""])
    changes = summary["paired"]["multi_vs_single"]["changed_answers"]
    if not changes:
        lines.append("No answers changed.")
    for item in changes:
        lines.append(
            f"- `{item['sample_id']}`: Single `{item['right_answer']}` -> "
            f"Multi `{item['left_answer']}`; reference `{item['reference_answer']}`."
        )
    lines.extend([
        "",
        "## Integrity",
        "",
    ])
    for name, value in summary["integrity"].items():
        lines.append(f"- `{name}`: `{str(value).lower()}`")
    lines.extend([
        "",
        "## Scope",
        "",
        "This is a 20-sample development comparison, not a paper-level statistical conclusion. "
        "Direct and Visual perception outputs are frozen from the cited baseline evidence; "
        "Multi Agent adds newly executed Reasoning and Verification calls.",
        "",
    ])
    return "\n".join(lines)


def _audit(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    cost_fields = (
        "agent_calls",
        "tool_calls",
        "model_calls",
        "model_input_tokens",
        "model_output_tokens",
        "message_text_tokens",
        "message_structure_bytes",
        "message_latent_bytes",
    )
    checks = {
        "sample_count": len(records) == expected_count,
        "unique_sample_ids": len({item["sample_id"] for item in records}) == len(records),
        "all_arms_present": all(
            set(item["arms"]) == set(ARM_NAMES) for item in records
        ),
        "all_arms_error_free": all(
            item["arms"][arm].get("error") is None
            for item in records
            for arm in ARM_NAMES
        ),
        "single_workspace_complete": all(
            item["arms"]["single_agent"]["result"]["workspace"]["version"] == 2
            for item in records
        ),
        "multi_workspace_complete": all(
            item["arms"]["multi_agent"]["result"]["workspace"]["version"] == 3
            for item in records
        ),
        "multi_path_exact": all(
            item["arms"]["multi_agent"]["result"]["agent_path"]
            == ["visual", "reasoning", "verification"]
            for item in records
        ),
        "multi_two_routed_messages": all(
            len(item["arms"]["multi_agent"]["result"]["messages"]) == 2
            for item in records
        ),
        "multi_four_transitions": all(
            len(item["arms"]["multi_agent"]["result"]["transitions"]) == 4
            for item in records
        ),
        "multi_label_isolation": all(
            set(item["arms"]["multi_agent"]["result"]["task"]["metadata"])
            == {"dataset_sample_id"}
            for item in records
        ),
        "multi_cost_matches_runtime": all(
            all(
                float(item["arms"]["multi_agent"]["cost"][field])
                == float(item["arms"]["multi_agent"]["result"]["total_cost"][field])
                for field in cost_fields
            )
            for item in records
        ),
        "answer_ownership_valid": all(
            any(
                candidate["created_by"] == "reasoning"
                and candidate["status"] == "proposed"
                for candidate in item["arms"]["multi_agent"]["result"]["workspace"]["answer_candidates"]
            )
            and any(
                candidate["created_by"] == "verification"
                and candidate["status"] == "verified"
                for candidate in item["arms"]["multi_agent"]["result"]["workspace"]["answer_candidates"]
            )
            for item in records
        ),
        "workspace_contains_no_binary_values": all(
            not any(
                marker in json.dumps(
                    item["arms"][arm]["result"]["workspace"],
                    ensure_ascii=False,
                )
                for marker in ("artifact_payload", "pixel_values", "tensor_values")
            )
            for item in records
            for arm in ("single_agent", "multi_agent")
        ),
    }
    return {
        "schema_version": "evors.phase4-comparison-audit.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def _write_checksums(output_dir: Path, names: Iterable[str]) -> None:
    files = {}
    for name in names:
        path = output_dir / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    _atomic_json(
        output_dir / "checksums.json",
        {
            "schema_version": "evors.evidence-checksums.v1",
            "algorithm": "sha256",
            "files": files,
        },
    )


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run(
    manifest: Path,
    model_config: Path,
    baseline_events: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 20260831,
) -> dict[str, Any]:
    samples = _load_samples(manifest)
    if limit is not None:
        samples = samples[:limit]
    baseline = _baseline_index(baseline_events)
    missing = [
        (sample["sample_id"], arm)
        for sample in samples
        for arm in ("direct_vlm", "single_agent")
        if (sample["sample_id"], arm) not in baseline
    ]
    if missing:
        raise ValueError(f"baseline evidence is incomplete: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    existing = _jsonl(events_path)
    ids = [item.get("sample_id") for item in existing]
    if len(ids) != len(set(ids)):
        raise ValueError("events.jsonl contains duplicate sample IDs")
    expected_ids = {sample["sample_id"] for sample in samples}
    unknown = set(ids) - expected_ids
    if unknown:
        raise ValueError(f"events.jsonl contains samples outside this run: {sorted(unknown)}")

    provenance = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "baseline_events": str(baseline_events),
        "baseline_events_sha256": _sha256(baseline_events),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    _atomic_json(output_dir / "run_config.json", provenance)
    backend = _backend(model_config, seed)
    completed = set(ids)
    with events_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            if sample["sample_id"] in completed:
                continue
            direct_record = baseline[(sample["sample_id"], "direct_vlm")]
            perception_record = baseline[(sample["sample_id"], "single_agent")]
            with Image.open(sample["image_ref"]) as opened:
                image = opened.convert("RGB")
            record = {
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"],
                "question": sample["question"],
                "reference_answer": sample["reference_answer"],
                "arms": {
                    "direct_vlm": _direct_arm(direct_record),
                    "single_agent": _safe_arm(
                        lambda: _single_arm(sample, image, perception_record)
                    ),
                    "multi_agent": _safe_arm(
                        lambda: _multi_arm(sample, image, perception_record, backend)
                    ),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            existing.append(record)
            completed.add(sample["sample_id"])
            checkpoint = {
                "schema_version": "evors.phase4-comparison-checkpoint.v1",
                "completed_sample_ids": sorted(completed),
                "remaining_sample_ids": [
                    item["sample_id"]
                    for item in samples
                    if item["sample_id"] not in completed
                ],
            }
            _atomic_json(output_dir / "checkpoint.json", checkpoint)
            partial = _summary(
                existing,
                expected_count=len(samples),
                provenance=provenance,
            )
            _atomic_json(output_dir / "summary.json", partial)
            print(
                json.dumps(
                    {
                        "sample": sample["sample_id"],
                        "direct": record["arms"]["direct_vlm"]["answer"],
                        "single": record["arms"]["single_agent"]["answer"],
                        "multi": record["arms"]["multi_agent"]["answer"],
                        "reference": sample["reference_answer"],
                        "multi_correct": record["arms"]["multi_agent"]["correct"],
                        "multi_error": record["arms"]["multi_agent"].get("error"),
                        "completed": len(completed),
                        "total": len(samples),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    records = _jsonl(events_path)
    summary = _summary(records, expected_count=len(samples), provenance=provenance)
    _atomic_json(output_dir / "summary.json", summary)
    _write_csvs(output_dir, records, summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    _atomic_json(output_dir / "audit.json", _audit(records, len(samples)))
    _write_checksums(
        output_dir,
        (
            "events.jsonl",
            "checkpoint.json",
            "summary.json",
            "results.csv",
            "accuracy_cost.csv",
            "REPORT.md",
            "run_config.json",
            "audit.json",
        ),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "samples": summary["completed_sample_count"],
                "accuracy": {
                    arm: summary["arms"][arm]["accuracy"] for arm in ARM_NAMES
                },
                "paired": summary["paired"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--baseline-events", type=Path, default=DEFAULT_BASELINE_EVENTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args(argv)
    summary = run(
        args.manifest,
        args.model_config,
        args.baseline_events,
        args.output_dir,
        limit=args.limit,
        seed=args.seed,
    )
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
