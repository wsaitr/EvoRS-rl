#!/usr/bin/env python3
"""Independent evidence validator for the EvoRS Phase 5 mechanism gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rs_hmcomm.control import LearnedRoutingPolicy  # noqa: E402
from rs_hmcomm.rl import EvoRSAgentLoop, RoutingRewardManager  # noqa: E402
from rs_hmcomm.rl.fixtures import (  # noqa: E402
    Phase5FixtureRunner,
    build_phase5_fixture_splits,
)


REQUIRED = (
    "REPORT.md",
    "audit.json",
    "checkpoint.json",
    "checksums.json",
    "events.jsonl",
    "run_config.json",
    "summary.json",
    "verl_eval.jsonl",
    "verl_train.jsonl",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate(directory: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 5 evidence files: {missing}")
    expected_checksums = _json(directory / "checksums.json")
    for name, expected in expected_checksums.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch: {name}")

    summary = _json(directory / "summary.json")
    audit = _json(directory / "audit.json")
    config = _json(directory / "run_config.json")
    events = _jsonl(directory / "events.jsonl")
    checkpoint = LearnedRoutingPolicy.from_checkpoint(directory / "checkpoint.json")
    if summary["schema_version"] != "evors.phase5-summary.v1":
        raise ValueError("unsupported Phase 5 summary schema")
    if audit.get("passed") is not True or not all(audit["checks"].values()):
        raise ValueError("persisted Phase 5 audit did not pass")
    if checkpoint.parameter_digest() != summary["parameter_sha256_after"]:
        raise ValueError("checkpoint digest does not match summary")
    if checkpoint.trained_steps != summary["trained_steps"]:
        raise ValueError("checkpoint training step count does not match summary")
    if summary["parameter_sha256_before"] == summary["parameter_sha256_after"]:
        raise ValueError("summary reports no parameter change")

    train_events = [item for item in events if item["split"] == "train"]
    eval_events = [item for item in events if item["split"] == "eval"]
    initial_events = [item for item in events if item["split"] == "initial_eval"]
    expected_train = int(config["epochs"]) * len(config["train_sample_ids"])
    if len(train_events) != expected_train:
        raise ValueError("training event count does not match config")
    if len(eval_events) != len(config["eval_sample_ids"]):
        raise ValueError("eval event count does not match config")
    if len(initial_events) != len(config["eval_sample_ids"]):
        raise ValueError("initial eval event count does not match config")
    if any(item["optimizer_update"] is None for item in train_events):
        raise ValueError("training event is missing optimizer update")
    if any(
        item["parameter_sha256_before"] == item["parameter_sha256_after"]
        for item in train_events
    ):
        raise ValueError("training event has no parameter change")
    if any(
        item["parameter_sha256_before"] != item["parameter_sha256_after"]
        for item in eval_events
    ):
        raise ValueError("eval event mutated parameters")
    for item in (*train_events, *eval_events, *initial_events):
        if item["causality"]["all_actions_precede_transitions"] is not True:
            raise ValueError("event failed action causality")
        if item["causality"]["linked_action_count"] != len(item["action_trace"]):
            raise ValueError("event action/transition count mismatch")

    _, eval_samples = build_phase5_fixture_splits()
    reproduced = {}
    reproduced_actions = {}
    for sample in eval_samples:
        rollout = EvoRSAgentLoop(
            checkpoint,
            Phase5FixtureRunner(max_steps=int(config["environment"]["max_steps"])),
            reward_manager=RoutingRewardManager(),
        ).run(
            sample,
            episode_id=f"phase5-validation-{sample.sample_id}",
            training=False,
            explore=False,
        )
        reproduced[sample.task.task_type] = rollout.runtime_result["agent_path"]
        reproduced_actions[sample.task.task_type] = [
            step["action_id"] for step in rollout.action_trace
        ]
    if reproduced != summary["held_out_paths_by_task"]:
        raise ValueError("checkpoint does not reproduce persisted greedy paths")
    if reproduced_actions != summary["held_out_actions_by_task"]:
        raise ValueError("checkpoint does not reproduce persisted greedy actions")

    for name in ("verl_train.jsonl", "verl_eval.jsonl"):
        records = _jsonl(directory / name)
        for record in records:
            reference = record["reward_model"]["ground_truth"]
            serialized_task = json.dumps(
                record["extra_info"]["task"], ensure_ascii=False, sort_keys=True
            )
            if reference in serialized_task:
                raise ValueError("reference answer leaked into verl task observation")

    return {
        "schema_version": "evors.phase5-validation.v1",
        "passed": True,
        "evidence_dir": str(directory.resolve()),
        "verified_files": len(REQUIRED),
        "training_events": len(train_events),
        "eval_events": len(eval_events),
        "checkpoint_sha256": checkpoint.parameter_digest(),
        "reproduced_paths": reproduced,
        "reproduced_actions": reproduced_actions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "evidence" / "2026-09-03_phase5_routing",
    )
    args = parser.parse_args(argv)
    result = validate(args.evidence.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
