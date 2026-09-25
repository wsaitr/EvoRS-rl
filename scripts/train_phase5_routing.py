#!/usr/bin/env python3
"""Train and audit the real Phase 5 centralized routing policy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rs_hmcomm.control import LearnedRoutingPolicy  # noqa: E402
from rs_hmcomm.rl import (  # noqa: E402
    EvoRSAgentLoop,
    RoutingPolicyTrainer,
    RoutingRewardManager,
    RoutingRewardWeights,
    RoutingTrainerConfig,
    VerlRoutingAdapter,
    summarize_events,
    verl_environment_status,
)
from rs_hmcomm.rl.fixtures import (  # noqa: E402
    Phase5FixtureRunner,
    build_phase5_fixture_splits,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _initial_gate(policy, runner, reward_manager, samples):
    events = []
    for sample in samples:
        loop = EvoRSAgentLoop(policy, runner, reward_manager=reward_manager)
        rollout = loop.run(
            sample,
            episode_id=f"phase5-initial-{sample.sample_id}",
            training=True,
            explore=False,
        )
        events.append(
            {
                "schema_version": "evors.routing-initial-eval.v1",
                "split": "initial_eval",
                "sample_id": sample.sample_id,
                "task_type": sample.task.task_type,
                "reference_answer": sample.reference_answer,
                "answer": rollout.answer,
                "correct": rollout.reward["task_score"] == 1.0,
                "agent_path": rollout.runtime_result["agent_path"],
                "action_trace": list(rollout.action_trace),
                "causality": dict(rollout.causality),
                "reward": dict(rollout.reward),
                "total_cost": rollout.runtime_result["total_cost"],
            }
        )
    return events


def _paths_by_task(events: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in events:
        result[item["task_type"]] = list(item["agent_path"])
    return result


def _actions_by_task(events: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in events:
        result[item["task_type"]] = [
            step["action_id"] for step in item["action_trace"]
        ]
    return result


def _checksums(directory: Path, names: Sequence[str]) -> dict[str, str]:
    values = {}
    for name in names:
        values[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return values


def run(config_path: Path, *, output_override: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "evors.phase5-config.v1":
        raise ValueError("unsupported Phase 5 config schema")
    output_dir = output_override or ROOT / raw["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(raw["seed"])
    trainer_config = RoutingTrainerConfig(
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        baseline_rate=float(raw["baseline_rate"]),
        entropy_coefficient=float(raw["entropy_coefficient"]),
        max_gradient=float(raw["max_gradient"]),
        seed=seed,
    )
    reward_weights = RoutingRewardWeights(**raw["reward"])
    reward_manager = RoutingRewardManager(reward_weights)
    runner = Phase5FixtureRunner(max_steps=int(raw["environment"]["max_steps"]))
    train_samples, eval_samples = build_phase5_fixture_splits()

    initial_policy = LearnedRoutingPolicy(seed=seed)
    initial_digest = initial_policy.parameter_digest()
    initial_events = _initial_gate(
        initial_policy,
        runner,
        reward_manager,
        eval_samples,
    )
    if initial_policy.parameter_digest() != initial_digest:
        raise RuntimeError("initial evaluation mutated the untrained policy")

    policy = LearnedRoutingPolicy(seed=seed)
    trainer = RoutingPolicyTrainer(
        policy,
        runner,
        reward_manager=reward_manager,
        config=trainer_config,
    )
    training = trainer.train_and_evaluate(train_samples, eval_samples)
    trained_summary = summarize_events(training.train_events)
    eval_summary = summarize_events(training.eval_events)
    initial_summary = summarize_events(initial_events)

    learned_paths = _paths_by_task(training.eval_events)
    learned_actions = _actions_by_task(training.eval_events)
    distinct_paths = {tuple(path) for path in learned_paths.values()}
    checks = {
        "parameter_changed": (
            training.initial_parameter_sha256 != training.final_parameter_sha256
        ),
        "experts_frozen": training.experts_frozen,
        "all_train_actions_causal": all(
            event["causality"]["all_actions_precede_transitions"]
            for event in training.train_events
        ),
        "all_eval_actions_causal": all(
            event["causality"]["all_actions_precede_transitions"]
            for event in training.eval_events
        ),
        "eval_did_not_update": all(
            event["parameter_sha256_before"] == event["parameter_sha256_after"]
            for event in training.eval_events
        ),
        "held_out_accuracy_one": eval_summary["accuracy"] == 1.0,
        "learned_reward_improves_initial": (
            eval_summary["mean_reward"] > initial_summary["mean_reward"]
        ),
        "task_conditioned_paths": len(distinct_paths) >= 2,
        "count_skips_knowledge": all(
            "knowledge" not in event["agent_path"]
            for event in training.eval_events
            if event["task_type"] == "object_quantity"
        ),
        "classification_uses_knowledge": all(
            "knowledge" in event["agent_path"]
            for event in training.eval_events
            if event["task_type"] == "airport_classification"
        ),
        "count_uses_text_structure": all(
            "dispatch_reasoning_text_structure"
            in [step["action_id"] for step in event["action_trace"]]
            and "dispatch_reasoning_hybrid"
            not in [step["action_id"] for step in event["action_trace"]]
            for event in training.eval_events
            if event["task_type"] == "object_quantity"
        ),
        "existence_uses_paid_hybrid": all(
            "dispatch_reasoning_hybrid"
            in [step["action_id"] for step in event["action_trace"]]
            and event["reward"]["message_cost"] > 0.0
            for event in training.eval_events
            if event["task_type"] == "object_existence"
        ),
        "existence_skips_knowledge": all(
            "knowledge" not in event["agent_path"]
            for event in training.eval_events
            if event["task_type"] == "object_existence"
        ),
        "task_conditioned_reasoning_modality": len(
            {
                step["action_id"]
                for event in training.eval_events
                for step in event["action_trace"]
                if step["action_id"].startswith("dispatch_reasoning_")
            }
        ) >= 2,
        "reference_absent_from_task": all(
            sample.reference_answer
            not in json.dumps(sample.task.to_dict(), ensure_ascii=False, sort_keys=True)
            for sample in (*train_samples, *eval_samples)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 5 mechanism gate failed: {failed}")

    checkpoint_path = policy.save(output_dir / "checkpoint.json")
    VerlRoutingAdapter.write_jsonl(train_samples, output_dir / "verl_train.jsonl")
    VerlRoutingAdapter.write_jsonl(eval_samples, output_dir / "verl_eval.jsonl")
    _write_jsonl(
        output_dir / "events.jsonl",
        [*initial_events, *training.train_events, *training.eval_events],
    )
    summary = {
        "schema_version": "evors.phase5-summary.v1",
        "claim_scope": "deterministic mechanism gate; not a paper accuracy result",
        "initial_eval": initial_summary,
        "train": trained_summary,
        "held_out_eval": eval_summary,
        "held_out_paths_by_task": learned_paths,
        "held_out_actions_by_task": learned_actions,
        "parameter_sha256_before": training.initial_parameter_sha256,
        "parameter_sha256_after": training.final_parameter_sha256,
        "trained_steps": policy.trained_steps,
        "experts_frozen": training.experts_frozen,
        "verl_environment": verl_environment_status(),
        "checkpoint": checkpoint_path.name,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "audit.json", {"checks": checks, "passed": True})
    _write_json(
        output_dir / "run_config.json",
        {
            **raw,
            "resolved_config_path": str(config_path.resolve()),
            "resolved_output_dir": str(output_dir.resolve()),
            "trainer": asdict(trainer_config),
            "reward": asdict(reward_weights),
            "train_sample_ids": [item.sample_id for item in train_samples],
            "eval_sample_ids": [item.sample_id for item in eval_samples],
        },
    )
    report = f"""# EvoRS Phase 5 Routing Mechanism Gate

- Scope: deterministic control-learning mechanism gate, not a paper benchmark.
- Initial held-out reward: {initial_summary['mean_reward']:.6f}
- Learned held-out reward: {eval_summary['mean_reward']:.6f}
- Held-out accuracy: {eval_summary['accuracy']:.3f}
- Initial mean Agent calls: {initial_summary['mean_agent_calls']:.3f}
- Learned mean Agent calls: {eval_summary['mean_agent_calls']:.3f}
- Learned paths: {json.dumps(eval_summary['paths'], ensure_ascii=False, sort_keys=True)}
- Learned actions by task: {json.dumps(learned_actions, ensure_ascii=False, sort_keys=True)}
- Policy parameters changed: yes
- Cognitive Expert fingerprints unchanged: yes
- Every sampled action linked to its pre-state transition: yes
- Evaluation parameter updates: zero
- verl installed in this environment: {summary['verl_environment']['installed']}

The learned greedy policy skips Knowledge for quantity/existence tasks, keeps
Knowledge for airport classification, uses Text+Structure for counting, and
uses paid Hybrid latent disclosure for existence. References are available only
to the reward environment, never to Controller or Expert observations.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    names = (
        "REPORT.md",
        "audit.json",
        "checkpoint.json",
        "events.jsonl",
        "run_config.json",
        "summary.json",
        "verl_eval.jsonl",
        "verl_train.jsonl",
    )
    _write_json(output_dir / "checksums.json", _checksums(output_dir, names))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "phase5_routing.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = run(args.config.resolve(), output_override=args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
