"""Run the Phase 1 Generalist with a mock, in-process, or local-server VLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import yaml

from rs_hmcomm.agents.generalist import GeneralistAgent
from rs_hmcomm.backends.mock import MockBackend
from rs_hmcomm.contracts import (
    BudgetLimits,
    EvaluationTarget,
    EventType,
    exact_match_score,
    normalize_short_answer,
)
from rs_hmcomm.datasets.manifest import iter_manifest_dataset
from rs_hmcomm.evolution import EvolutionBudget, EvolutionConfig
from rs_hmcomm.observability import EventJsonlLogger
from rs_hmcomm.runtime import SingleAgentRuntime
from rs_hmcomm.tools import CropTool, ToolRegistry


DEFAULT_CONFIG = Path("configs/experiments/phase1_generalist.yaml")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: Path, *, expected_phase: int = 1) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    values = _mapping(config, "config")
    allowed = {"phase", "dataset", "backend", "runtime", "budget", "logging"}
    if expected_phase == 2:
        allowed.add("evolution")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"unknown Phase {expected_phase} config fields: {sorted(unknown)}"
        )
    if values.get("phase") != expected_phase:
        raise ValueError(
            f"Phase {expected_phase} runner requires phase: {expected_phase}"
        )
    return values


def _build_evolution_config(config: Mapping[str, Any]) -> EvolutionConfig:
    values = _mapping(config.get("evolution"), "evolution")
    allowed = {
        "enabled",
        "population_size",
        "max_generations",
        "no_progress_generations",
        "min_fitness_gain",
        "seed",
        "budget",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown evolution config fields: {sorted(unknown)}")
    budget_values = _mapping(values.get("budget"), "evolution.budget")
    budget_unknown = set(budget_values) - {
        "max_candidate_evaluations",
        "max_variations",
    }
    if budget_unknown:
        raise ValueError(
            f"unknown evolution budget fields: {sorted(budget_unknown)}"
        )
    return EvolutionConfig(
        enabled=values.get("enabled", True),
        population_size=values.get("population_size", 2),
        max_generations=values.get("max_generations", 1),
        no_progress_generations=values.get("no_progress_generations", 1),
        min_fitness_gain=values.get("min_fitness_gain", 0.0),
        seed=values.get("seed", 0),
        budget=EvolutionBudget(
            max_candidate_evaluations=budget_values.get(
                "max_candidate_evaluations",
                2,
            ),
            max_variations=budget_values.get("max_variations", 1),
        ),
    )


def build_backend(
    values: dict[str, Any],
    *,
    backend_override: str | None,
    model_path_override: str | None,
    device_override: str | None,
    allow_download: bool,
    model_override: str | None = None,
    base_url_override: str | None = None,
    api_key_env_override: str | None = None,
):
    backend_type = backend_override or values.get("type")
    if backend_type == "mock":
        return MockBackend(), "mock_control_flow_only"
    if backend_type == "openai_compatible":
        model = model_override or values.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("backend.model or --model is required for openai_compatible")
        base_url = base_url_override or values.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "backend.base_url or --base-url is required for openai_compatible"
            )
        api_key_env = api_key_env_override or values.get("api_key_env")
        api_key = None
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not api_key_env.strip():
                raise ValueError("backend.api_key_env must be null or a non-empty name")
            api_key = os.environ.get(api_key_env)
            if api_key is None:
                raise ValueError(f"API key environment variable is not set: {api_key_env}")
        extra_body = values.get("extra_body", {})
        if not isinstance(extra_body, dict):
            raise ValueError("backend.extra_body must be a mapping")
        from rs_hmcomm.backends.openai_compatible import OpenAICompatibleVLMBackend

        return (
            OpenAICompatibleVLMBackend(
                model,
                base_url=base_url,
                api_key=api_key,
                timeout=values.get("timeout_seconds", 120),
                max_tokens=values.get("max_tokens", 256),
                extra_body=extra_body,
                enable_thinking=values.get("enable_thinking", False),
                preserve_thinking=values.get("preserve_thinking", False),
            ),
            "real_vlm_strict_gate",
        )
    if backend_type != "qwen_vl":
        raise ValueError(
            "backend.type must be mock, qwen_vl, or openai_compatible"
        )

    model_path = model_path_override or values.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("a local --model-path is required for qwen_vl")
    from rs_hmcomm.backends.qwen_vl import Qwen25VLBackend

    return (
        Qwen25VLBackend(
            model_path,
            revision=values.get("revision"),
            device=device_override or values.get("device", "auto"),
            local_files_only=not allow_download,
            max_new_tokens=values.get("max_new_tokens", 256),
            min_pixels=values.get("min_pixels"),
            max_pixels=values.get("max_pixels"),
        ),
        "in_process_vlm_integration_only",
    )


def _strict_expected_model_id(
    backend_values: Mapping[str, Any],
    *,
    model_override: str | None,
) -> str | None:
    """Return the exact server model identifier required by the strict gate."""

    if backend_values.get("type") != "openai_compatible" and model_override is None:
        return None
    value = model_override or backend_values.get("model")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _counterfactual_fixture_failures(samples: Sequence[Any]) -> list[str]:
    """Validate that live samples form one image-dependent counterfactual pair."""

    failures: list[str] = []
    if len(samples) < 2:
        return ["strict live gate requires at least two counterfactual samples"]

    groups: list[str] = []
    image_digests: list[str] = []
    for sample in samples:
        metadata = sample.metadata or {}
        group = metadata.get("counterfactual_group")
        if not isinstance(group, str) or not group.strip():
            failures.append(
                f"sample {sample.sample_id} is missing metadata.counterfactual_group"
            )
        else:
            groups.append(group.strip())
        pixel_digest = hashlib.sha256(
            sample.image.mode.encode("ascii")
            + repr(sample.image.size).encode("ascii")
            + sample.image.tobytes()
        ).hexdigest()
        image_digests.append(pixel_digest)
        if metadata.get("pixel_sha256") != pixel_digest:
            failures.append(
                f"sample {sample.sample_id} pixel SHA-256 does not match its manifest"
            )

    if groups and len(set(groups)) != 1:
        failures.append("strict live samples must belong to one counterfactual_group")
    if len({sample.question for sample in samples}) != 1:
        failures.append("counterfactual samples must use the exact same question")
    if len({sample.answer for sample in samples}) < 2:
        failures.append("counterfactual samples must have different reference answers")
    if len(set(image_digests)) != len(image_digests):
        failures.append("counterfactual samples must contain distinct image pixels")
    return failures


def _phase1_causal_failures(
    run_result: Any,
    *,
    expected_model_id: str,
    require_tool: bool,
) -> list[str]:
    """Apply Phase-1-specific causal, provenance, model and usage checks."""

    failures: list[str] = []
    try:
        run_result.validate_phase1_completion(require_tool=require_tool)
    except (TypeError, ValueError) as exc:
        failures.append(f"Phase 1 causal validation failed: {type(exc).__name__}: {exc}")

    events = run_result.trace.events
    model_events = [event for event in events if event.event_type == EventType.MODEL_CALLED]
    if run_result.model_id != expected_model_id:
        failures.append(
            f"server model_id mismatch: expected {expected_model_id!r}, "
            f"observed {run_result.model_id!r}"
        )

    for event in model_events:
        event_model_id = event.action.get("model_id")
        if event_model_id != expected_model_id:
            failures.append(
                f"model event {event.event_id} identifies {event_model_id!r}, "
                f"expected {expected_model_id!r}"
            )
        if event.result.get("response_model_reported") is not True:
            failures.append(
                f"model event {event.event_id} did not receive an explicit server model_id"
            )
        if event.result.get("response_model_id") != expected_model_id:
            failures.append(
                f"model event {event.event_id} server-reported model_id "
                f"{event.result.get('response_model_id')!r}, expected {expected_model_id!r}"
            )
        if event.result.get("token_accounting") != "measured":
            failures.append(f"model event {event.event_id} has non-measured token usage")
        usage = event.result.get("usage")
        if not isinstance(usage, Mapping):
            failures.append(f"model event {event.event_id} has no usage object")
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens <= 0
        ):
            failures.append(f"model event {event.event_id} has no positive input tokens")
        if (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens <= 0
        ):
            failures.append(f"model event {event.event_id} has no positive output tokens")
        if event.cost_delta.model_input_tokens != input_tokens:
            failures.append(f"model event {event.event_id} input token ledger mismatch")
        if event.cost_delta.model_output_tokens != output_tokens:
            failures.append(f"model event {event.event_id} output token ledger mismatch")
    return failures


def _phase2_evolution_failures(run_result: Any) -> list[str]:
    """Verify that private answer evolution materially participated in a run."""

    failures: list[str] = []
    events = run_result.trace.events
    proposed = [
        event for event in events if event.event_type == EventType.CANDIDATE_PROPOSED
    ]
    evaluated = [
        event for event in events if event.event_type == EventType.CANDIDATE_EVALUATED
    ]
    selected = [
        event for event in events if event.event_type == EventType.CANDIDATE_SELECTED
    ]
    if len(proposed) < 2:
        failures.append("Phase 2 requires at least one seed and one variation")
    if len(evaluated) < 2:
        failures.append("Phase 2 requires fitness for at least two candidates")
    if not selected:
        failures.append("Phase 2 has no candidate selection event")
    zero_cost = type(run_result.cost)()
    if any(
        event.cost_delta != zero_cost for event in proposed + evaluated + selected
    ):
        failures.append("candidate bookkeeping events must not own physical cost")

    proposed_ids = {event.action.get("candidate_id") for event in proposed}
    for event in selected:
        if event.result.get("selected_candidate_id") not in proposed_ids:
            failures.append("selected candidate was not proposed in this Agent run")

    summary = run_result.agent_result.metadata.get("answer_evolution")
    if not isinstance(summary, Mapping):
        failures.append("Agent result is missing the answer_evolution summary")
    elif run_result.answer is not None:
        selected_payload = summary.get("selected_candidate")
        if not isinstance(selected_payload, Mapping):
            failures.append("evolution summary is missing the selected candidate")
        else:
            payload = selected_payload.get("payload")
            if not isinstance(payload, Mapping) or payload.get("answer") != run_result.answer:
                failures.append("selected private candidate did not determine the public answer")

    public_answers = [
        record
        for event in events
        if event.event_type == EventType.RECORD_ADDED
        for record in event.result.get("records", ())
        if record.get("record_type") == "answer_candidate"
    ]
    if len(public_answers) != 1:
        failures.append("Phase 2 must commit exactly one public answer candidate")
    return failures


def run(
    config_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
    backend_override: str | None = None,
    model_path_override: str | None = None,
    device_override: str | None = None,
    allow_download: bool = False,
    model_override: str | None = None,
    base_url_override: str | None = None,
    api_key_env_override: str | None = None,
    expected_phase: int = 1,
) -> dict[str, Any]:
    if expected_phase not in {1, 2}:
        raise ValueError("expected_phase must be 1 or 2")
    config = load_config(config_path, expected_phase=expected_phase)
    dataset_values = _mapping(config.get("dataset"), "dataset")
    manifest_value = dataset_values.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("dataset.manifest must be a non-empty path")
    samples = tuple(
        iter_manifest_dataset(
            manifest_value,
            limit=dataset_values.get("limit"),
        )
    )
    if not samples:
        raise ValueError("Phase 1 manifest contains no samples")

    backend_values = _mapping(config.get("backend"), "backend")
    backend, validation_scope = build_backend(
        backend_values,
        backend_override=backend_override,
        model_path_override=model_path_override,
        device_override=device_override,
        allow_download=allow_download,
        model_override=model_override,
        base_url_override=base_url_override,
        api_key_env_override=api_key_env_override,
    )
    evolution_config = (
        _build_evolution_config(config) if expected_phase == 2 else None
    )
    if expected_phase == 2 and validation_scope == "mock_control_flow_only":
        validation_scope = "mock_evolution_mechanism_only"
    runtime_values = _mapping(config.get("runtime"), "runtime")
    require_tool = bool(runtime_values.get("require_tool", False))
    strict_live_gate = validation_scope == "real_vlm_strict_gate"
    expected_model_id = _strict_expected_model_id(
        backend_values,
        model_override=model_override,
    )
    preflight_failures: list[str] = []
    if strict_live_gate:
        if expected_model_id is None:
            preflight_failures.append("strict live gate has no exact expected model_id")
        preflight_failures.extend(_counterfactual_fixture_failures(samples))
    runtime = SingleAgentRuntime(
        GeneralistAgent(backend),
        ToolRegistry((CropTool(),)),
        BudgetLimits.from_dict(_mapping(config.get("budget"), "budget")),
        max_steps=runtime_values.get("max_steps", 4),
        evolution_config=evolution_config,
    )

    if output_path is None:
        logging_values = _mapping(config.get("logging"), "logging")
        raw_output = logging_values.get("output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("logging.output must be a non-empty path")
        output_path = Path(raw_output)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("logging output must be a file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite without --overwrite: {output_path}")
    summary_path = Path(f"{output_path}.summary.json")
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite summary without --overwrite: {summary_path}"
        )
    logger = EventJsonlLogger(output_path, append=False)

    results: list[dict[str, Any]] = []
    all_completed = True
    all_gate_checks_passed = not preflight_failures
    for sample in samples:
        is_mock = validation_scope == "mock_control_flow_only"
        result = runtime.run(
            sample.image,
            sample.question,
            sample_id=sample.sample_id,
            image_ref=f"manifest://{Path(manifest_value).name}/{sample.sample_id}",
            task_type=sample.task_type,
            require_tool=require_tool,
            task_metadata=(
                {"mock_answer": sample.answer, "fixture": True}
                if is_mock
                else {"fixture": False}
            ),
        )
        logger.write_trace(result.trace)
        score = (
            exact_match_score(
                result.answer,
                EvaluationTarget(
                    sample_id=sample.sample_id,
                    references=(sample.answer,),
                    task_type=sample.task_type,
                ),
            )
            if result.answer is not None
            else None
        )
        completed = result.stop_reason.value == "completed"
        all_completed = all_completed and completed
        gate_failures: list[str] = list(preflight_failures) if strict_live_gate else []
        if strict_live_gate:
            if not completed:
                gate_failures.append(
                    f"episode did not complete: {result.stop_reason.value}"
                )
            if expected_model_id is None:
                gate_failures.append("strict live gate has no exact expected model_id")
            else:
                gate_failures.extend(
                    _phase1_causal_failures(
                        result,
                        expected_model_id=expected_model_id,
                        require_tool=require_tool,
                    )
                )
            if score is None or not score.correct:
                gate_failures.append("answer does not exactly match the visual reference")
        if expected_phase == 2:
            gate_failures.extend(_phase2_evolution_failures(result))
        gate_passed = completed and not gate_failures
        all_gate_checks_passed = all_gate_checks_passed and gate_passed
        results.append(
            {
                "sample_id": sample.sample_id,
                "completed": completed,
                "answer": result.answer,
                "normalized_answer": (
                    normalize_short_answer(result.answer)
                    if result.answer is not None
                    else None
                ),
                "reference_used_for_evaluation_only": sample.answer,
                "normalized_reference_used_for_evaluation_only": (
                    normalize_short_answer(sample.answer)
                ),
                "score": score.to_dict() if score is not None else None,
                "stop_reason": result.stop_reason.value,
                "cost": result.cost.to_dict(),
                "wall_time_ms": result.wall_time_ms,
                "model_id": result.model_id,
                "model_revision": result.model_revision,
                "evidence_ids": [
                    item.evidence_id for item in result.agent_result.evidence
                ],
                "gate_passed": (
                    gate_passed if strict_live_gate or expected_phase == 2 else None
                ),
                "gate_failures": gate_failures,
            }
        )

    passed = (
        all_gate_checks_passed
        if strict_live_gate or expected_phase == 2
        else all_completed
    )
    summary = {
        "schema_version": f"evors.phase{expected_phase}-gate-summary.v1",
        "phase": expected_phase,
        "status": "passed" if passed else "failed",
        "validation_scope": validation_scope,
        "scientific_claim": False,
        "gate": {
            "strict": strict_live_gate,
            "evolution_mechanism_checked": expected_phase == 2,
            "passed": passed,
            "expected_model_id": expected_model_id if strict_live_gate else None,
            "require_tool": require_tool,
            "counterfactual_sample_count": len(samples) if strict_live_gate else 0,
            "preflight_failures": preflight_failures,
            "requirements": (
                ([
                    "counterfactual_images",
                    "exact_match_per_sample",
                    "exact_server_model_id",
                    "measured_positive_tokens_per_model_call",
                    "phase1_tool_evidence_answer_structural_trace",
                ] + (["private_candidate_variation_fitness_selection"] if expected_phase == 2 else []))
                if strict_live_gate
                else (
                    ["private_candidate_variation_fitness_selection"]
                    if expected_phase == 2
                    else ["control_flow_only"]
                )
            ),
            "evidence_semantics": "structural_provenance_only",
        },
        "sample_count": len(results),
        "event_log": str(output_path),
        "summary_file": str(summary_path),
        "artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "manifest_path": str(manifest_value),
            "manifest_sha256": _sha256_file(Path(manifest_value)),
            "event_log_sha256": _sha256_file(output_path),
        },
        "results": results,
    }
    temporary_summary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary_path.replace(summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("mock", "qwen_vl", "openai_compatible"),
        default=None,
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="name of an environment variable containing the API key",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        args.config,
        output_path=args.output,
        overwrite=args.overwrite,
        backend_override=args.backend,
        model_path_override=args.model_path,
        device_override=args.device,
        allow_download=args.allow_download,
        model_override=args.model,
        base_url_override=args.base_url,
        api_key_env_override=args.api_key_env,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
