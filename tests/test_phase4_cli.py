from __future__ import annotations

import json


def test_phase4_cli_runs_all_baselines_and_persists_replay(tmp_path, capsys) -> None:
    from scripts.run_phase4_collaboration import main

    output = tmp_path / "phase4-summary.json"
    exit_code = main(["--output", str(output)])
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["phase"] == 4
    assert summary["scientific_claim"] is False
    assert summary["langgraph_runtime"] is True
    assert summary["policies"] == ["fixed", "all_agent", "rule"]
    assert summary["run_count"] == 6
    assert all(item["completed"] for item in summary["results"])
    assert all(
        item["agent_path"] == item["replay_agent_path"]
        for item in summary["results"]
    )
    assert all(item["transitions"] for item in summary["results"])
    assert json.loads(output.read_text(encoding="utf-8")) == summary
