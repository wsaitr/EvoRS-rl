from __future__ import annotations

import json

from rs_hmcomm.contracts import EventType
from rs_hmcomm.observability import load_event_traces


def test_phase2_mock_cli_runs_private_evolution_mechanism(tmp_path, capsys) -> None:
    from scripts.run_phase2_evolution import main

    output = tmp_path / "phase2-events.jsonl"
    exit_code = main(["--output", str(output)])
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["phase"] == 2
    assert summary["status"] == "passed"
    assert summary["validation_scope"] == "mock_evolution_mechanism_only"
    assert summary["scientific_claim"] is False
    assert summary["results"][0]["answer"] == "6"
    assert summary["results"][0]["cost"]["model_calls"] == 3
    assert summary["results"][0]["gate_passed"] is True

    trace = load_event_traces(output)[0]
    event_types = [event.event_type for event in trace.events]
    assert event_types.count(EventType.CANDIDATE_PROPOSED) == 2
    assert event_types.count(EventType.CANDIDATE_EVALUATED) == 2
    assert event_types.count(EventType.CANDIDATE_SELECTED) == 1
    trace.validate_complete()
