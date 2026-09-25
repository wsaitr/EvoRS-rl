"""Deterministic Phase 0 contract, metric, budget, and replay smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from rs_hmcomm.contracts import (
    AgentObservation,
    AgentResult,
    AgentResultStatus,
    AnswerCandidate,
    Assignment,
    BudgetLimits,
    Claim,
    CostVector,
    DatasetSpec,
    EpisodeEvent,
    EpisodeTrace,
    EvaluationTarget,
    EventType,
    Evidence,
    TaskContext,
    canonical_manifest_digest,
    deterministic_sample_ids,
    exact_match_score,
)
from rs_hmcomm.datasets.manifest import iter_manifest_dataset
from rs_hmcomm.observability import EventJsonlLogger, load_event_traces


DEFAULT_CONFIG = Path("configs/experiments/phase0.yaml")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    config = _mapping(loaded, "config")
    unknown = set(config) - {"phase", "seed", "dataset", "budget", "logging"}
    if unknown:
        raise ValueError(f"unknown Phase 0 config fields: {sorted(unknown)}")
    if config.get("phase") != 0:
        raise ValueError("Phase 0 validator requires phase: 0")
    return config


def build_trace(
    config: dict[str, Any],
) -> tuple[EpisodeTrace, AgentResult, str, str, dict[str, Any]]:
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    dataset_values = _mapping(config.get("dataset"), "dataset")
    unknown_dataset_fields = set(dataset_values) - {
        "name",
        "revision",
        "split",
        "limit",
        "manifest",
    }
    if unknown_dataset_fields:
        raise ValueError(f"unknown dataset fields: {sorted(unknown_dataset_fields)}")
    manifest_value = dataset_values.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("dataset.manifest must be a non-empty path")
    manifest_path = Path(manifest_value).resolve(strict=True)
    manifest_records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"blank manifest line at {line_number}")
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            manifest_records.append(record)
    manifest_digest = canonical_manifest_digest(manifest_records)
    dataset = DatasetSpec(
        name=dataset_values.get("name"),
        revision=dataset_values.get("revision"),
        split=dataset_values.get("split"),
        seed=seed,
        limit=dataset_values.get("limit"),
        manifest_digest=manifest_digest,
    )
    budget = BudgetLimits.from_dict(_mapping(config.get("budget"), "budget"))
    samples = tuple(iter_manifest_dataset(manifest_path))
    if not samples:
        raise ValueError("dataset manifest must contain at least one sample")
    selected = deterministic_sample_ids(
        (sample.sample_id for sample in samples),
        seed=dataset.seed,
        limit=dataset.limit,
    )
    sample_id = selected[0]
    sample = next(item for item in samples if item.sample_id == sample_id)
    raw_record = next(item for item in manifest_records if item["sample_id"] == sample_id)
    image_path = (manifest_path.parent / raw_record["image"]).resolve(strict=True)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    declared_image_sha256 = (sample.metadata or {}).get("image_sha256")
    if image_sha256 != declared_image_sha256:
        raise ValueError("fixture image SHA-256 does not match manifest metadata")
    episode_id = f"phase0-{seed}-{sample_id}"

    task = TaskContext(
        task_id=sample_id,
        image_ref=f"manifest://{dataset.name}/{sample_id}",
        question=sample.question,
        task_type=sample.task_type,
        answer_format="short_text",
        metadata={
            "dataset": dataset.name,
            "revision": dataset.revision,
            "split": dataset.split,
            "seed": dataset.seed,
            "manifest_digest": dataset.manifest_digest,
            "fixture_scope": "contracts_only",
        },
    )
    assignment = Assignment(
        assignment_id="assignment-generalist-0",
        agent_id="generalist",
        objective="Produce an evidence-backed short answer",
    )
    observation = AgentObservation(
        observation_id="observation-0",
        agent_id="generalist",
        task=task,
        assignment=assignment,
        step=0,
        budget_limits=budget,
    )
    evidence = Evidence(
        evidence_id="evidence-visual-0",
        kind="visual_fixture",
        summary=f"Synthetic contract fixture label is {sample.answer}",
        created_by="generalist",
        artifact_refs=(task.image_ref,),
        confidence=1.0,
    )
    claim = Claim(
        claim_id="claim-count-0",
        statement=f"The synthetic fixture answer is {sample.answer}",
        created_by="generalist",
        evidence_ids=(evidence.evidence_id,),
        confidence=1.0,
    )
    answer = AnswerCandidate(
        candidate_id="answer-0",
        answer=sample.answer,
        created_by="generalist",
        claim_ids=(claim.claim_id,),
        evidence_ids=(evidence.evidence_id,),
        confidence=1.0,
    )
    cost = CostVector(
        agent_calls=1,
        model_calls=1,
        model_input_tokens=32,
        model_output_tokens=8,
        billed_compute_ms=1.0,
    )
    result = AgentResult(
        result_id="result-generalist-0",
        agent_id="generalist",
        assignment_id=assignment.assignment_id,
        status=AgentResultStatus.COMPLETED,
        evidence=(evidence,),
        claims=(claim,),
        answer_candidates=(answer,),
    )
    result.validate_against(observation)
    if AgentObservation.from_dict(observation.to_dict()) != observation:
        raise AssertionError("AgentObservation JSON round-trip failed")
    if AgentResult.from_dict(result.to_dict()) != result:
        raise AssertionError("AgentResult JSON round-trip failed")

    events = (
        EpisodeEvent(
            episode_id=episode_id,
            event_id="event-0",
            seq=0,
            event_type=EventType.EPISODE_STARTED,
            actor="runtime",
            action={"task_id": task.task_id},
            result={
                "observation_id": observation.observation_id,
                "budget_limits": budget.to_dict(),
            },
            elapsed_ms=0.0,
        ),
        EpisodeEvent(
            episode_id=episode_id,
            event_id="event-1",
            seq=1,
            event_type=EventType.AGENT_DISPATCHED,
            actor="runtime",
            action={"agent_id": "generalist", "assignment_id": assignment.assignment_id},
            result={"status": "dispatched"},
            cost_delta=CostVector(agent_calls=1),
            elapsed_ms=0.1,
        ),
        EpisodeEvent(
            episode_id=episode_id,
            event_id="event-2",
            seq=2,
            event_type=EventType.MODEL_CALLED,
            actor="contract-fixture",
            action={"model_id": "contract-fixture"},
            result={"token_accounting": "estimated", "fixture": True},
            cost_delta=CostVector(
                model_calls=cost.model_calls,
                model_input_tokens=cost.model_input_tokens,
                model_output_tokens=cost.model_output_tokens,
                billed_compute_ms=cost.billed_compute_ms,
            ),
            elapsed_ms=0.5,
        ),
        EpisodeEvent(
            episode_id=episode_id,
            event_id="event-3",
            seq=3,
            event_type=EventType.RECORD_ADDED,
            actor="runtime",
            action={"result_id": result.result_id},
            result={
                "records": [
                    {"record_id": evidence.evidence_id, "record_type": "evidence"},
                    {"record_id": claim.claim_id, "record_type": "claim"},
                    {"record_id": answer.candidate_id, "record_type": "answer_candidate"},
                ]
            },
            elapsed_ms=0.8,
        ),
        EpisodeEvent(
            episode_id=episode_id,
            event_id="event-4",
            seq=4,
            event_type=EventType.EPISODE_FINISHED,
            actor="runtime",
            action={"answer_candidate_ref": answer.candidate_id},
            result={"stop_reason": "completed"},
            elapsed_ms=1.0,
        ),
    )
    trace = EpisodeTrace.replay(events, budget_limits=budget)
    replay_state = trace.validate_complete()
    budget.require_within(trace.total_cost, wall_time_ms=replay_state.wall_time_ms)
    fixture_info = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "image_sha256": image_sha256,
        "image_size": list(sample.image.size),
        "synthetic": True,
    }
    return trace, result, sample_id, sample.answer, fixture_info


def run(
    config_path: Path,
    output_path: Path | None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    trace, result, sample_id, reference_answer, fixture_info = build_trace(config)
    if output_path is None:
        logging_values = _mapping(config.get("logging"), "logging")
        raw_output = logging_values.get("output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("logging.output must be a non-empty path")
        output_path = Path(raw_output)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("logging output must be a file, not a directory")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing event log without --overwrite: {output_path}"
        )

    logger = EventJsonlLogger(output_path, append=False)
    logger.write_trace(trace)
    replayed = load_event_traces(output_path)
    if replayed != (trace,):
        raise AssertionError("JSONL replay does not reproduce the original trace")
    replayed[0].validate_complete()

    answer = result.answer_candidates[0].answer
    score = exact_match_score(
        answer,
        EvaluationTarget(
            sample_id=sample_id,
            references=(reference_answer,),
            task_type="synthetic_counting",
        ),
    )
    canonical_trace = json.dumps(
        trace.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "phase": 0,
        "status": "contract_smoke_passed",
        "scope": "contracts_only_no_visual_model",
        "sample_id": sample_id,
        "fixture": fixture_info,
        "seed": config["seed"],
        "event_count": len(trace.events),
        "task_score": score.to_dict(),
        "cost": trace.total_cost.to_dict(),
        "token_accounting": "estimated_fixture",
        "trace_sha256": hashlib.sha256(canonical_trace).hexdigest(),
        "output": str(output_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.config, args.output, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
