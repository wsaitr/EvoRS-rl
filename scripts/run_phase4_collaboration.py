"""Run Phase 4 fixed/all/rule collaboration baselines on one LangGraph runtime."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from rs_hmcomm.contracts import BudgetLimits, StopReason, TaskContext
from rs_hmcomm.control import (
    AllAgentPolicy,
    EvoRSController,
    FixedWorkflowPolicy,
    RuleRoutingPolicy,
)
from rs_hmcomm.experts import (
    KnowledgeExpert,
    ReasoningExpert,
    VerificationExpert,
    VisualExpert,
)
from rs_hmcomm.runtime import LangGraphMultiAgentRuntime


DEFAULT_CONFIG = Path("configs/experiments/phase4_collaboration.yaml")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = _mapping(yaml.safe_load(handle), "config")
    unknown = set(values) - {"phase", "tasks", "policies", "runtime", "budget", "logging"}
    if unknown:
        raise ValueError(f"unknown Phase 4 config fields: {sorted(unknown)}")
    if values.get("phase") != 4:
        raise ValueError("Phase 4 runner requires phase: 4")
    return values


def _experts():
    return {
        "visual": VisualExpert(),
        "reasoning": ReasoningExpert(),
        "verification": VerificationExpert(),
        "knowledge": KnowledgeExpert(),
    }


_POLICIES = {
    "fixed": FixedWorkflowPolicy,
    "all_agent": AllAgentPolicy,
    "rule": RuleRoutingPolicy,
}


def _write_summary(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("Phase 4 output must be a file")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(
    config_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    raw_tasks = config.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a non-empty list")
    raw_policies = config.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("policies must be a non-empty list")
    if any(item not in _POLICIES for item in raw_policies):
        raise ValueError(f"policies must use names from {sorted(_POLICIES)}")
    if len(raw_policies) != len(set(raw_policies)):
        raise ValueError("policies must not contain duplicates")

    runtime_values = _mapping(config.get("runtime"), "runtime")
    budget_values = _mapping(config.get("budget"), "budget")
    budget = BudgetLimits.from_dict(budget_values)
    max_steps = runtime_values.get("max_steps", 8)

    results: list[dict[str, Any]] = []
    for raw_task in raw_tasks:
        task_values = _mapping(raw_task, "task")
        task = TaskContext(
            task_id=task_values["task_id"],
            image_ref=task_values["image_ref"],
            question=task_values["question"],
            task_type=task_values.get("task_type", "scripted_phase4"),
            answer_format="short_text",
            metadata={
                "scripted_fixture": True,
                "mock_answer": str(task_values["mock_answer"]),
                "scientific_claim": False,
            },
        )
        for policy_name in raw_policies:
            policy = _POLICIES[policy_name]()
            result = LangGraphMultiAgentRuntime(
                EvoRSController(policy),
                _experts(),
                budget,
                max_steps=max_steps,
            ).run(
                task,
                episode_id=f"episode-{task.task_id}-{policy_name}",
            )
            replay = result.replay()
            results.append(
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "policy": policy_name,
                    "policy_class": result.policy,
                    "runtime": result.runtime,
                    "completed": result.stop_reason == StopReason.COMPLETED,
                    "stop_reason": result.stop_reason.value,
                    "answer": result.answer,
                    "agent_path": list(result.agent_path),
                    "workspace_version": result.workspace.version,
                    "message_count": len(result.messages),
                    "messages": [item.to_dict() for item in result.messages],
                    "transitions": [item.to_dict() for item in result.transitions],
                    "cost": result.total_cost.to_dict(),
                    "replay_agent_path": list(replay.agent_path),
                    "replay_message_ids": list(replay.disclosed_message_ids),
                    "answer_candidate_ref": result.answer_candidate_ref,
                }
            )

    count_rule = next(
        item
        for item in results
        if item["task_id"] == "phase4-counting" and item["policy"] == "rule"
    )
    type_rule = next(
        item
        for item in results
        if item["task_id"] == "phase4-airport-type" and item["policy"] == "rule"
    )
    failures: list[str] = []
    if any(not item["completed"] for item in results):
        failures.append("at least one collaboration baseline did not complete")
    if "knowledge" in count_rule["agent_path"]:
        failures.append("rule counting route activated Knowledge Expert")
    if not {"visual", "reasoning", "knowledge"}.issubset(type_rule["agent_path"]):
        failures.append("rule airport-type route missed a required Expert")
    if any(item["agent_path"] != item["replay_agent_path"] for item in results):
        failures.append("replayed Agent path differs from executed path")
    if any(item["runtime"] != "langgraph" for item in results):
        failures.append("a baseline did not execute through LangGraph")

    if output_path is None:
        logging_values = _mapping(config.get("logging"), "logging")
        raw_output = logging_values.get("output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("logging.output must be a non-empty path")
        output_path = Path(raw_output)
    summary = {
        "schema_version": "evors.phase4-collaboration-summary.v1",
        "phase": 4,
        "status": "passed" if not failures else "failed",
        "validation_scope": "scripted_collaboration_mechanism",
        "scientific_claim": False,
        "langgraph_runtime": True,
        "policies": list(raw_policies),
        "shared_budget": budget.to_dict(),
        "failures": failures,
        "run_count": len(results),
        "results": results,
        "output": str(output_path),
    }
    _write_summary(output_path, summary, overwrite=overwrite)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.config, output_path=args.output, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
