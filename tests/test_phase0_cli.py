from __future__ import annotations

import json
from pathlib import Path

from rs_hmcomm.observability import load_event_traces


def test_phase0_validator_writes_replayable_trace(tmp_path, capsys) -> None:
    from scripts.validate_phase0 import main

    output = tmp_path / "phase0.jsonl"
    assert main(["--config", "configs/experiments/phase0.yaml", "--output", str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "contract_smoke_passed"
    assert summary["scope"] == "contracts_only_no_visual_model"
    assert summary["event_count"] == 5
    assert summary["task_score"]["correct"] is True
    assert summary["cost"]["agent_calls"] == 1
    assert len(summary["trace_sha256"]) == 64
    assert Path(summary["output"]) == output
    assert len(load_event_traces(output)) == 1
