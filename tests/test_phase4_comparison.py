from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from rs_hmcomm.backends.base import AgentOutput, BackendUsage, TokenAccounting


class _QueueBackend:
    model_name = "fake-paired-vlm"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate_turn(self, request):
        if not self.responses:
            raise AssertionError("unexpected model call")
        return AgentOutput(
            text=self.responses.pop(0),
            usage=BackendUsage(
                input_tokens=20,
                output_tokens=4,
                latency_ms=5.0,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id=self.model_name,
        )


def _baseline_record(
    sample: dict,
    arm: str,
    answer: str,
    observations: list[dict],
    *,
    decision_rule: str,
    tool_calls: int,
) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "task_type": sample["task_type"],
        "question": sample["question"],
        "reference_answer": sample["reference_answer"],
        "arm": arm,
        "answer": answer,
        "model_calls": len(observations),
        "tool_calls": tool_calls,
        "input_tokens": sum(item["input_tokens"] for item in observations),
        "output_tokens": sum(item["output_tokens"] for item in observations),
        "wall_time_ms": sum(item["latency_ms"] for item in observations),
        "model_ids": ["real-recorded-vlm"],
        "observations": observations,
        "decision_rule": decision_rule,
        "correct": True,
        "error": None,
    }


def _observation(view: str, answer: str) -> dict:
    return {
        "view": view,
        "answer": answer,
        "input_tokens": 10,
        "output_tokens": 2,
        "latency_ms": 5.0,
        "model_id": "real-recorded-vlm",
    }


def test_paired_comparison_persists_resumable_complete_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import run_phase4_comparison as module

    count_image = tmp_path / "count.png"
    existence_image = tmp_path / "existence.png"
    Image.new("RGB", (64, 64), "white").save(count_image)
    Image.new("RGB", (64, 64), "black").save(existence_image)
    samples = [
        {
            "sample_id": "count-1",
            "image_ref": str(count_image),
            "task_type": "object_quantity",
            "question": "How many aircraft are visible?",
            "reference_answer": "2",
        },
        {
            "sample_id": "exist-1",
            "image_ref": str(existence_image),
            "task_type": "object_existence",
            "question": "Is there vegetation?",
            "reference_answer": "yes",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(item) for item in samples) + "\n",
        encoding="utf-8",
    )
    count_direct = [_observation("whole_image", "2")]
    count_agent = [
        _observation("whole_image", "2"),
        _observation(
            "object_localization",
            '{"objects":[{"bbox":[0.1,0.1,0.2,0.2]},'
            '{"bbox":[0.5,0.5,0.6,0.6]}]}',
        ),
        _observation("enhanced_global_2x", "2"),
    ]
    existence = [_observation("whole_image", "Yes")]
    baseline_rows = [
        _baseline_record(samples[0], "direct", "2", count_direct, decision_rule="single_whole_image_call", tool_calls=0),
        _baseline_record(samples[0], "agent", "2", count_agent, decision_rule="global_scale_consensus", tool_calls=1),
        _baseline_record(samples[1], "direct", "Yes", existence, decision_rule="single_whole_image_call", tool_calls=0),
        _baseline_record(samples[1], "agent", "Yes", existence, decision_rule="whole_image_sufficient", tool_calls=0),
    ]
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        "\n".join(json.dumps(item) for item in baseline_rows) + "\n",
        encoding="utf-8",
    )
    backend = _QueueBackend(
        [
            "2",
            '{"verdict":"accept","revised_answer":"2","confidence":0.8}',
            "Yes",
            '{"verdict":"accept","revised_answer":"Yes","confidence":0.8}',
        ]
    )
    monkeypatch.setattr(module, "_backend", lambda config, seed: backend)
    output_dir = tmp_path / "evidence"

    summary = module.run(
        manifest,
        Path("configs/models/qwen38_27b_q4_local.yaml"),
        baseline,
        output_dir,
    )

    assert summary["status"] == "completed"
    assert summary["completed_sample_count"] == 2
    assert all(summary["arms"][arm]["accuracy"] == 1.0 for arm in module.ARM_NAMES)
    assert summary["by_task_type"]["object_quantity"]["multi_agent"]["correct"] == 1
    assert summary["by_task_type"]["object_existence"]["multi_agent"]["correct"] == 1
    assert len(backend.responses) == 0
    for name in (
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "results.csv",
        "accuracy_cost.csv",
        "REPORT.md",
        "run_config.json",
        "audit.json",
        "checksums.json",
    ):
        assert (output_dir / name).is_file()
    events = [
        json.loads(line)
        for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 2
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert all(audit["checks"].values())
    for event in events:
        multi = event["arms"]["multi_agent"]
        assert multi["result"]["agent_path"] == ["visual", "reasoning", "verification"]
        assert "reference_answer" not in multi["result"]["task"]["metadata"]
        assert multi["result"]["transitions"]
        assert multi["result"]["messages"]

    # A second invocation is a resume/no-op and must not duplicate evidence.
    second = module.run(
        manifest,
        Path("configs/models/qwen38_27b_q4_local.yaml"),
        baseline,
        output_dir,
    )
    assert second["status"] == "completed"
    assert len((output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 2
