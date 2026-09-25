#!/usr/bin/env python3
"""Produce a fail-closed completion audit for EvoRS Phases 0--5.

The audit deliberately separates deterministic mechanism gates from the small
real-Qwen development gates.  It does not promote either into a Phase 6 paper
result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "2026-09-03_phase0_5_completion_audit"
EXPECTED_MODEL = "evors/qwen3.8-27b-q4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _check_declared_checksums(directory: Path) -> tuple[bool, str]:
    declared = _json(directory / "checksums.json")
    entries = declared.get("files", declared) if isinstance(declared, dict) else {}
    mismatches = []
    for name, expected in entries.items():
        if isinstance(expected, dict):
            expected_hash = expected.get("sha256")
            expected_bytes = expected.get("bytes")
        else:
            expected_hash = expected
            expected_bytes = None
        path = directory / name
        if (
            not path.is_file()
            or _sha256(path) != expected_hash
            or (expected_bytes is not None and path.stat().st_size != expected_bytes)
        ):
            mismatches.append(name)
    return not mismatches, "verified" if not mismatches else f"mismatch: {mismatches}"


def _reasoning_transition(event: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        item
        for item in event.get("transitions", [])
        if item.get("active_expert") == "reasoning"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one reasoning transition, found {len(matches)}")
    return matches[0]


class CompletionAudit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, phase: int | str, name: str, passed: bool, detail: Any) -> None:
        self.checks.append(
            {"phase": phase, "name": name, "passed": bool(passed), "detail": detail}
        )

    def phase0(self) -> None:
        events = _jsonl(ROOT / "outputs" / "completion_audit" / "phase0")
        expected = [
            "episode_started",
            "agent_dispatched",
            "model_called",
            "record_added",
            "episode_finished",
        ]
        actual = [item.get("event_type") for item in events]
        self.check(0, "event_contract", actual == expected, actual)
        self.check(
            0,
            "schema_sequence_and_completion",
            all(item.get("schema_version") == "evors.event.v1" for item in events)
            and [item.get("seq") for item in events] == list(range(len(events)))
            and events[-1].get("result", {}).get("stop_reason") == "completed",
            {"event_count": len(events), "final": events[-1].get("result")},
        )

    def phase1(self) -> None:
        mechanism = _json(
            ROOT
            / "outputs"
            / "completion_audit"
            / "phase1"
            / "events.jsonl.summary.json"
        )
        self.check(
            1,
            "generalist_end_to_end_mechanism",
            mechanism.get("status") == "passed"
            and mechanism.get("gate", {}).get("require_tool") is True
            and mechanism.get("results", [{}])[0].get("completed") is True,
            {
                "status": mechanism.get("status"),
                "validation_scope": mechanism.get("validation_scope"),
            },
        )

        directory = ROOT / "evidence" / "2026-09-01_single_agent_perception_gate"
        summary = _json(directory / "summary.json")
        events = _jsonl(directory / "events.jsonl")
        by_arm: dict[str, list[Mapping[str, Any]]] = {"direct": [], "agent": []}
        for event in events:
            if event.get("arm") in by_arm:
                by_arm[event["arm"]].append(event)
        paired = (
            len(events) == 40
            and len(by_arm["direct"]) == len(by_arm["agent"]) == 20
            and {item["sample_id"] for item in by_arm["direct"]}
            == {item["sample_id"] for item in by_arm["agent"]}
        )
        direct_correct = sum(bool(item.get("correct")) for item in by_arm["direct"])
        agent_correct = sum(bool(item.get("correct")) for item in by_arm["agent"])
        all_real_model = all(
            item.get("error") is None
            and set(item.get("model_ids", [])) == {EXPECTED_MODEL}
            for item in events
        )
        manifest = ROOT / summary["manifest"]
        model_config = ROOT / summary["model_config"]
        provenance_valid = (
            _sha256(manifest) == summary.get("manifest_sha256")
            and _sha256(model_config) == summary.get("model_config_sha256")
        )
        self.check(
            1,
            "real_qwen_paired_gate_integrity",
            summary.get("schema_version") == "evors.single-agent-perception-gate.v1"
            and summary.get("scientific_claim") is False
            and summary.get("agent_errors") == summary.get("direct_errors") == 0
            and paired
            and all_real_model
            and provenance_valid,
            {
                "events": len(events),
                "model": EXPECTED_MODEL,
                "provenance_valid": provenance_valid,
            },
        )
        self.check(
            1,
            "single_agent_exceeds_direct_vlm",
            direct_correct == summary.get("direct_correct") == 10
            and agent_correct == summary.get("agent_correct") == 12
            and agent_correct > direct_correct,
            {"direct": direct_correct, "single_agent": agent_correct, "samples": 20},
        )

    def phase2(self) -> None:
        mechanism = _json(
            ROOT
            / "outputs"
            / "completion_audit"
            / "phase2"
            / "events.jsonl.summary.json"
        )
        self.check(
            2,
            "candidate_variation_fitness_selection_mechanism",
            mechanism.get("status") == "passed"
            and mechanism.get("gate", {}).get("evolution_mechanism_checked") is True,
            mechanism.get("gate", {}),
        )
        events = _jsonl(
            ROOT / "evidence" / "2026-09-02_phase4_comparison" / "events.jsonl"
        )
        traces = []
        for event in events:
            evolution = (
                event.get("arms", {})
                .get("single_agent", {})
                .get("result", {})
                .get("evolution")
            )
            generations = (
                evolution.get("private_trace", {}).get("generations", [])
                if evolution
                else []
            )
            if generations:
                traces.append((event.get("sample_id"), generations))
        valid = any(
            all(
                generation.get("population")
                and generation.get("variations")
                and generation.get("fitness_evaluations")
                and generation.get("selection")
                for generation in generations
            )
            for _, generations in traces
        )
        self.check(
            2,
            "real_perception_linked_private_evolution_trace",
            valid,
            {"samples_with_generations": [item[0] for item in traces]},
        )

    def phase3(self) -> None:
        summary = _json(
            ROOT / "outputs" / "completion_audit" / "phase3" / "summary.json"
        )
        item = summary.get("results", [{}])[0]
        self.check(
            3,
            "workspace_hierarchy_artifactstore_latent_mechanism",
            summary.get("status") == "passed"
            and item.get("latent_consumed") is True
            and item.get("latent_bytes_loaded", 0) > 0
            and item.get("workspace_contains_binary_payload") is False
            and item.get("unrelated_record_hidden") is True
            and set(item.get("spatial_levels", [])) == {"scene", "object"},
            {
                "levels": item.get("spatial_levels"),
                "latent_bytes": item.get("latent_bytes_loaded"),
                "workspace_binary": item.get("workspace_contains_binary_payload"),
            },
        )

    def phase4(self) -> None:
        mechanism = _json(
            ROOT / "outputs" / "completion_audit" / "phase4" / "summary.json"
        )
        self.check(
            4,
            "controller_router_expert_pool_mechanism",
            mechanism.get("status") == "passed"
            and mechanism.get("langgraph_runtime") is True
            and set(mechanism.get("policies", [])) == {"fixed", "all_agent", "rule"}
            and all(item.get("completed") for item in mechanism.get("results", []))
            and all(
                item.get("agent_path") == item.get("replay_agent_path")
                for item in mechanism.get("results", [])
            ),
            {"runs": mechanism.get("run_count"), "policies": mechanism.get("policies")},
        )

        directory = ROOT / "evidence" / "2026-09-02_phase4_comparison"
        summary = _json(directory / "summary.json")
        checksum_ok, checksum_detail = _check_declared_checksums(directory)
        arms = summary.get("arms", {})
        self.check(
            4,
            "real_qwen_paired_three_arm_development_gate",
            summary.get("scientific_claim") is False
            and summary.get("status") == "completed"
            and summary.get("expected_sample_count")
            == summary.get("completed_sample_count")
            == 20
            and arms.get("direct_vlm", {}).get("correct") == 10
            and arms.get("single_agent", {}).get("correct") == 12
            and arms.get("multi_agent", {}).get("correct") == 14
            and all(summary.get("integrity", {}).values())
            and checksum_ok,
            {
                "correct": {
                    name: values.get("correct") for name, values in arms.items()
                },
                "checksums": checksum_detail,
                "claim_scope": "development gate only",
            },
        )

    def phase5(self) -> None:
        directory = ROOT / "evidence" / "2026-09-03_phase5_routing"
        summary = _json(directory / "summary.json")
        audit = _json(directory / "audit.json")
        checksum_ok, checksum_detail = _check_declared_checksums(directory)
        actions = summary.get("held_out_actions_by_task", {})
        self.check(
            5,
            "learned_policy_training_and_replay",
            audit.get("passed") is True
            and all(audit.get("checks", {}).values())
            and summary.get("trained_steps") == 720
            and summary.get("held_out_eval", {}).get("episodes") == 6
            and summary.get("held_out_eval", {}).get("accuracy") == 1.0
            and summary.get("parameter_sha256_before")
            != summary.get("parameter_sha256_after")
            and summary.get("experts_frozen") is True
            and checksum_ok,
            {
                "steps": summary.get("trained_steps"),
                "eval": summary.get("held_out_eval"),
                "checksums": checksum_detail,
            },
        )
        self.check(
            5,
            "task_conditioned_activation_and_modality",
            actions.get("object_quantity")
            == [
                "dispatch_visual",
                "dispatch_reasoning_text_structure",
                "dispatch_verification",
                "finish",
            ]
            and actions.get("airport_classification")
            == [
                "dispatch_visual",
                "dispatch_knowledge",
                "dispatch_reasoning_text_structure",
                "dispatch_verification",
                "finish",
            ]
            and actions.get("object_existence")
            == [
                "dispatch_visual",
                "dispatch_reasoning_hybrid",
                "dispatch_verification",
                "finish",
            ],
            actions,
        )

        text_event = _jsonl(
            ROOT
            / "evidence"
            / "2026-09-03_phase5_real_qwen_smoke"
            / "events.jsonl"
        )[0]
        text_reasoning = _reasoning_transition(text_event)
        latent_artifacts = [
            item
            for item in text_event.get("workspace", {}).get("artifacts", [])
            if item.get("kind") == "latent"
        ]
        self.check(
            5,
            "real_qwen_text_structure_denies_latent",
            text_event.get("correct") is True
            and text_reasoning.get("decision", {}).get("policy_action_id")
            == "dispatch_reasoning_text_structure"
            and text_event.get("total_cost", {}).get("message_latent_bytes") == 0
            and text_reasoning.get("expert_metrics", {}).get("latent_diagnostic_count") == 0
            and bool(latent_artifacts)
            and all(item.get("encoder") == "qwen_native_v1" for item in latent_artifacts),
            {
                "native_latent_artifacts": len(latent_artifacts),
                "communicated_latent_bytes": text_event.get("total_cost", {}).get(
                    "message_latent_bytes"
                ),
                "reasoning_latent_count": text_reasoning.get("expert_metrics", {}).get(
                    "latent_diagnostic_count"
                ),
            },
        )

        hybrid_event = _jsonl(
            ROOT
            / "evidence"
            / "2026-09-03_phase5_real_qwen_hybrid_smoke"
            / "events.jsonl"
        )[0]
        hybrid_reasoning = _reasoning_transition(hybrid_event)
        self.check(
            5,
            "real_qwen_hybrid_authorizes_and_charges_latent",
            hybrid_reasoning.get("decision", {}).get("policy_action_id")
            == "dispatch_reasoning_hybrid"
            and hybrid_event.get("total_cost", {}).get("message_latent_bytes", 0) > 0
            and hybrid_reasoning.get("expert_metrics", {}).get(
                "latent_diagnostic_count", 0
            )
            > 0,
            {
                "correct": hybrid_event.get("correct"),
                "communicated_latent_bytes": hybrid_event.get("total_cost", {}).get(
                    "message_latent_bytes"
                ),
                "reasoning_latent_count": hybrid_reasoning.get(
                    "expert_metrics", {}
                ).get("latent_diagnostic_count"),
            },
        )

    def paper_and_scope(self) -> None:
        source = ROOT / "doc" / "论文" / "1" / "EvoRS-Comm_论文源码.tex"
        pdf = ROOT / "doc" / "论文" / "1" / "EvoRS-Comm_论文源码.pdf"
        self.check(
            "paper",
            "source_and_pdf_present",
            source.is_file()
            and pdf.is_file()
            and pdf.stat().st_size > 10_000
            and pdf.read_bytes().startswith(b"%PDF"),
            {"source": str(source.relative_to(ROOT)), "pdf_bytes": pdf.stat().st_size},
        )

    def compose_configs(self, output_dir: Path) -> None:
        commands = [
            [
                "docker",
                "compose",
                "--env-file",
                "configs/docker/compose.defaults.env",
                "-f",
                name,
                "config",
                "--quiet",
            ]
            for name in (
                "docker-compose.yml",
                "docker-compose.pfs.yml",
                "docker-compose.npu.yml",
            )
        ]
        records = []
        passed = True
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                records.append(
                    {
                        "command": command,
                        "returncode": completed.returncode,
                        "output": completed.stdout,
                    }
                )
                passed = passed and completed.returncode == 0
            except FileNotFoundError as exc:
                records.append(
                    {"command": command, "returncode": None, "output": str(exc)}
                )
                passed = False
        _write_json(output_dir / "compose_validation.json", records)
        self.check(
            5,
            "cpu_pfs_npu_compose_contracts",
            passed,
            {"validated": [item["command"][-3] for item in records]},
        )
        self.check(
            6,
            "phase6_not_claimed",
            True,
            "Not executed: complete baselines, ablations, multiple seeds, and cost curves remain Phase 6.",
        )

    def run_tests(self, output_dir: Path) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output_dir / "pytest.log").write_text(completed.stdout, encoding="utf-8")
        tail = "\n".join(completed.stdout.strip().splitlines()[-3:])
        self.check("tests", "full_offline_suite", completed.returncode == 0, tail)


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "outputs" / "completion_audit" / "phase0",
        ROOT / "evidence" / "2026-09-01_single_agent_perception_gate" / "events.jsonl",
        ROOT / "evidence" / "2026-09-02_phase4_comparison" / "summary.json",
        ROOT / "evidence" / "2026-09-03_phase5_routing" / "checkpoint.json",
        ROOT / "evidence" / "2026-09-03_phase5_real_qwen_smoke" / "events.jsonl",
        ROOT
        / "evidence"
        / "2026-09-03_phase5_real_qwen_hybrid_smoke"
        / "events.jsonl",
        ROOT / "doc" / "论文" / "1" / "EvoRS-Comm_论文源码.pdf",
    )
    return {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path) for path in paths}


def run(output_dir: Path, *, run_tests: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = CompletionAudit()
    audit.phase0()
    audit.phase1()
    audit.phase2()
    audit.phase3()
    audit.phase4()
    audit.phase5()
    audit.paper_and_scope()
    audit.compose_configs(output_dir)
    if run_tests:
        audit.run_tests(output_dir)

    failures = [item for item in audit.checks if not item["passed"]]
    phases = {
        str(phase): all(
            item["passed"] for item in audit.checks if str(item["phase"]) == str(phase)
        )
        for phase in range(6)
    }
    summary = {
        "schema_version": "evors.phase0-5-completion-audit.v1",
        "status": "passed" if not failures and all(phases.values()) else "failed",
        "scientific_claim": False,
        "completed_scope": "Phase 0 through Phase 5 implementation and development gates",
        "excluded_scope": "Phase 6 paper-level experiments and statistical claims",
        "phases": phases,
        "checks": audit.checks,
        "failure_count": len(failures),
        "deployment": {
            "local_controller_training": "completed",
            "verl_token_policy_training": "not claimed; adapter only",
            "ascend_execution": "not claimed; target host execution evidence pending",
            "reason": "The current 7x21 controller is intentionally CPU-trained.",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "source_hashes.json", _source_hashes())
    rows = [
        "# EvoRS Phase 0--5 Completion Audit",
        "",
        f"- Status: **{summary['status']}**",
        "- Claim scope: implementation and development/mechanism gates only.",
        "- Scientific claim: false.",
        "- Phase 6: not executed and not claimed.",
        "",
        "## Checks",
        "",
    ]
    rows.extend(
        f"- [{'x' if item['passed'] else ' '}] Phase {item['phase']} — {item['name']}"
        for item in audit.checks
    )
    rows.extend(
        [
            "",
            "## Boundaries",
            "",
            "The 20-sample real-Qwen comparisons are reproducible development gates, not a paper benchmark. ",
            "The deterministic Phase 5 fixture proves causal policy learning and task-conditioned communication, not model accuracy. ",
            "A complete Phase 6 still requires registered baselines, ablations, multiple seeds, confidence intervals, and accuracy-cost curves.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")
    output_names = ["REPORT.md", "source_hashes.json", "summary.json"]
    output_names.append("compose_validation.json")
    if (output_dir / "pytest.log").is_file():
        output_names.append("pytest.log")
    _write_json(
        output_dir / "checksums.json",
        {name: _sha256(output_dir / name) for name in output_names},
    )
    if summary["status"] != "passed":
        names = ", ".join(item["name"] for item in failures)
        raise RuntimeError(f"Phase 0--5 completion audit failed: {names}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.output.resolve(), run_tests=args.run_tests)
    print(
        json.dumps(
            {
                "schema_version": summary["schema_version"],
                "status": summary["status"],
                "phases": summary["phases"],
                "checks": len(summary["checks"]),
                "failure_count": summary["failure_count"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
