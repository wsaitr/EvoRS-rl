from __future__ import annotations

import json


def test_phase3_runner_writes_reads_and_consumes_latent(tmp_path, capsys) -> None:
    from scripts.run_phase3_representation import main

    output = tmp_path / "phase3-summary.json"
    exit_code = main(["--output", str(output)])
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["phase"] == 3
    assert summary["scientific_claim"] is False
    assert summary["learned_visual_representation"] is False
    result = summary["results"][0]
    assert result["workspace_version"] == 1
    assert result["unrelated_record_hidden"] is True
    assert result["workspace_contains_binary_payload"] is False
    assert result["spatial_levels"] == ["scene", "object"]
    assert result["object_parent_level"] == "scene"
    assert result["latent_consumed"] is True
    assert result["latent_value_count"] == 12
    assert result["latent_bytes_loaded"] == 48
    assert result["cost"]["message_latent_bytes"] == 48
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_phase3_text_structure_path_runs_with_latent_disabled(tmp_path, capsys) -> None:
    from scripts.run_phase3_representation import main

    output = tmp_path / "phase3-no-latent.json"
    exit_code = main(["--output", str(output), "--no-latent"])
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    result = summary["results"][0]
    assert result["latent_enabled"] is False
    assert result["latent_consumed"] is False
    assert result["artifact_descriptor_count"] == 0
    assert result["cost"]["message_latent_bytes"] == 0
