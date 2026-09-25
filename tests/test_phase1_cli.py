from __future__ import annotations

import json

import pytest

from rs_hmcomm.observability import load_event_traces


def test_phase1_mock_cli_is_explicitly_non_scientific(tmp_path, capsys) -> None:
    from scripts.run_phase1_generalist import main

    output = tmp_path / "phase1-events.jsonl"
    exit_code = main(
        [
            "--config",
            "configs/experiments/phase1_generalist.yaml",
            "--output",
            str(output),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["validation_scope"] == "mock_control_flow_only"
    assert summary["scientific_claim"] is False
    assert summary["results"][0]["completed"] is True
    assert summary["results"][0]["cost"]["tool_calls"] == 1
    summary_path = output.with_name(output.name + ".summary.json")
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary == summary
    assert persisted_summary["artifacts"]["event_log_sha256"]
    assert persisted_summary["results"][0]["normalized_answer"]
    from scripts.validate_phase1_gate import validate_phase1_gate

    validation = validate_phase1_gate(summary_path)
    assert validation["status"] == "validated"
    assert validation["sample_count"] == 1
    traces = load_event_traces(output)
    assert len(traces) == 1
    traces[0].validate_complete()

    persisted_summary["results"][0]["answer"] = (
        f" {persisted_summary['results'][0]['answer']} "
    )
    summary_path.write_text(json.dumps(persisted_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized answer|answer differs"):
        validate_phase1_gate(summary_path)


def test_in_process_qwen_backend_is_not_named_a_strict_live_gate(monkeypatch) -> None:
    from rs_hmcomm.backends import qwen_vl
    from scripts.run_phase1_generalist import build_backend

    sentinel = object()
    monkeypatch.setattr(qwen_vl, "Qwen25VLBackend", lambda *args, **kwargs: sentinel)
    backend, scope = build_backend(
        {"type": "qwen_vl", "model_path": "local-model", "device": "cpu"},
        backend_override=None,
        model_path_override=None,
        device_override=None,
        allow_download=False,
    )

    assert backend is sentinel
    assert scope == "in_process_vlm_integration_only"
