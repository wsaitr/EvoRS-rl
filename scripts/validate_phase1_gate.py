"""Offline validation for a persisted Phase 1 gate summary and event log."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rs_hmcomm.contracts import (
    EvaluationTarget,
    EventType,
    exact_match_score,
    normalize_short_answer,
)
from rs_hmcomm.datasets.manifest import iter_manifest_dataset
from rs_hmcomm.observability import load_event_traces

if __package__:
    from scripts.run_phase1_generalist import load_config
else:
    from run_phase1_generalist import load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _records(trace: Any, record_type: str) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for event in trace.events:
        if event.event_type != EventType.RECORD_ADDED:
            continue
        values = event.result.get("records")
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if not isinstance(value, Mapping) or value.get("record_type") != record_type:
                continue
            record_id = value.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{record_type} event has no record_id")
            if record_id in records:
                raise ValueError(f"duplicate persisted record_id: {record_id}")
            records[record_id] = value
    return records


def validate_phase1_gate(summary_path: str | Path) -> dict[str, Any]:
    """Fail closed unless a persisted, passed Phase 1 run is self-consistent."""

    summary_file = Path(summary_path)
    raw_summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary = _mapping(raw_summary, "summary")
    if summary.get("schema_version") != "evors.phase1-gate-summary.v1":
        raise ValueError("unsupported Phase 1 summary schema_version")
    if summary.get("phase") != 1 or summary.get("status") != "passed":
        raise ValueError("offline validator requires a passed Phase 1 summary")
    if summary.get("scientific_claim") is not False:
        raise ValueError("Phase 1 integration evidence cannot be a scientific claim")

    artifacts = _mapping(summary.get("artifacts"), "summary.artifacts")
    config_path = _resolve_recorded_path(artifacts.get("config_path"), "config_path")
    manifest_path = _resolve_recorded_path(
        artifacts.get("manifest_path"), "manifest_path"
    )
    event_log_path = _resolve_recorded_path(summary.get("event_log"), "event_log")
    for path, digest_key in (
        (config_path, "config_sha256"),
        (manifest_path, "manifest_sha256"),
        (event_log_path, "event_log_sha256"),
    ):
        expected_digest = artifacts.get(digest_key)
        if not isinstance(expected_digest, str) or _sha256_file(path) != expected_digest:
            raise ValueError(f"persisted {digest_key} does not match {path}")

    config = load_config(config_path)
    dataset = _mapping(config.get("dataset"), "config.dataset")
    configured_manifest = dataset.get("manifest")
    if _resolve_recorded_path(configured_manifest, "dataset.manifest") != manifest_path:
        raise ValueError("summary manifest_path differs from the pinned config")
    samples = tuple(
        iter_manifest_dataset(manifest_path, limit=dataset.get("limit"))
    )
    samples_by_id = {sample.sample_id: sample for sample in samples}

    raw_results = summary.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("summary.results must be a list")
    if summary.get("sample_count") != len(raw_results) or len(raw_results) != len(samples):
        raise ValueError("summary sample_count does not match results/manifest")
    results_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_result in raw_results:
        result = _mapping(raw_result, "summary result")
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in results_by_id:
            raise ValueError("summary results contain a missing or duplicate sample_id")
        results_by_id[sample_id] = result
    if set(results_by_id) != set(samples_by_id):
        raise ValueError("summary sample IDs differ from the pinned manifest")

    traces = load_event_traces(event_log_path)
    traces_by_task: dict[str, Any] = {}
    for trace in traces:
        replay = trace.validate_complete()
        if replay.task_id in traces_by_task:
            raise ValueError(f"duplicate event trace for task {replay.task_id}")
        traces_by_task[replay.task_id] = (trace, replay)
    if set(traces_by_task) != set(samples_by_id):
        raise ValueError("event-log task IDs differ from the pinned manifest")

    gate = _mapping(summary.get("gate"), "summary.gate")
    strict = gate.get("strict") is True
    if gate.get("passed") is not True:
        raise ValueError("persisted gate is not marked passed")
    expected_model_id = gate.get("expected_model_id")
    if strict and (not isinstance(expected_model_id, str) or not expected_model_id):
        raise ValueError("strict gate has no expected_model_id")
    require_tool = gate.get("require_tool") is True

    for sample_id, sample in samples_by_id.items():
        result = results_by_id[sample_id]
        trace, replay = traces_by_task[sample_id]
        if result.get("completed") is not True or result.get("stop_reason") != "completed":
            raise ValueError(f"sample {sample_id} is not completed")
        if replay.stop_reason.value != "completed" or replay.budget_violations:
            raise ValueError(f"sample {sample_id} replay did not complete within budget")
        if result.get("cost") != trace.total_cost.to_dict():
            raise ValueError(f"sample {sample_id} summary cost differs from event log")
        if result.get("reference_used_for_evaluation_only") != sample.answer:
            raise ValueError(f"sample {sample_id} reference differs from manifest")

        answer = result.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"sample {sample_id} has no answer")
        if result.get("normalized_answer") != normalize_short_answer(answer):
            raise ValueError(f"sample {sample_id} normalized answer is inconsistent")
        if result.get(
            "normalized_reference_used_for_evaluation_only"
        ) != normalize_short_answer(sample.answer):
            raise ValueError(f"sample {sample_id} normalized reference is inconsistent")
        target = EvaluationTarget(
            sample_id=sample_id,
            references=(sample.answer,),
            task_type=sample.task_type,
        )
        expected_score = exact_match_score(answer, target).to_dict()
        if result.get("score") != expected_score:
            raise ValueError(f"sample {sample_id} score is not reproducible")
        if strict and expected_score["correct"] is not True:
            raise ValueError(f"strict sample {sample_id} is not exact-match correct")
        if strict and (result.get("gate_passed") is not True or result.get("gate_failures") != []):
            raise ValueError(f"strict sample {sample_id} did not persist a clean gate")

        answer_records = _records(trace, "answer_candidate")
        answer_ref = trace.events[-1].action.get("answer_candidate_ref")
        answer_record = answer_records.get(answer_ref)
        if answer_record is None:
            raise ValueError(f"sample {sample_id} answer record is missing")
        if answer_record.get("answer_sha256") != hashlib.sha256(answer.encode("utf-8")).hexdigest():
            raise ValueError(f"sample {sample_id} answer differs from the event-log digest")
        evidence_ids = result.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            raise ValueError(f"sample {sample_id} evidence_ids must be a list")
        persisted_evidence_ids = answer_record.get("evidence_ids")
        if not isinstance(persisted_evidence_ids, (list, tuple)) or list(
            persisted_evidence_ids
        ) != evidence_ids:
            raise ValueError(f"sample {sample_id} answer evidence differs from event log")
        if not set(evidence_ids).issubset(_records(trace, "evidence")):
            raise ValueError(f"sample {sample_id} answer references missing evidence")

        if require_tool and trace.total_cost.tool_calls < 1:
            raise ValueError(f"sample {sample_id} did not execute the required tool")
        if strict:
            model_events = [
                event for event in trace.events if event.event_type == EventType.MODEL_CALLED
            ]
            if not model_events:
                raise ValueError(f"sample {sample_id} has no model event")
            for event in model_events:
                usage = event.result.get("usage")
                if (
                    event.action.get("model_id") != expected_model_id
                    or event.result.get("response_model_reported") is not True
                    or event.result.get("response_model_id") != expected_model_id
                    or event.result.get("token_accounting") != "measured"
                    or not isinstance(usage, Mapping)
                    or not isinstance(usage.get("input_tokens"), int)
                    or usage["input_tokens"] <= 0
                    or not isinstance(usage.get("output_tokens"), int)
                    or usage["output_tokens"] <= 0
                    or event.cost_delta.model_input_tokens != usage["input_tokens"]
                    or event.cost_delta.model_output_tokens != usage["output_tokens"]
                ):
                    raise ValueError(
                        f"sample {sample_id} has an invalid strict model event"
                    )

    return {
        "status": "validated",
        "phase1_status": "passed",
        "validation_scope": summary.get("validation_scope"),
        "sample_count": len(samples),
        "summary_file": str(summary_file),
        "event_log": str(event_log_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args(argv)
    result = validate_phase1_gate(args.summary)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
