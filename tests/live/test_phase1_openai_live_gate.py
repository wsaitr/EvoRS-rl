from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.live_vlm
def test_openai_qwen_counterfactual_strict_gate(tmp_path) -> None:
    """Exercise the actual local OpenAI endpoint; skipped without explicit opt-in."""

    model = os.environ.get("EVORS_LIVE_OPENAI_MODEL")
    base_url = os.environ.get("EVORS_LIVE_OPENAI_BASE_URL")
    if not model or not base_url:
        pytest.skip(
            "set EVORS_LIVE_OPENAI_MODEL and EVORS_LIVE_OPENAI_BASE_URL "
            "to run the local Qwen strict gate"
        )

    from scripts.run_phase1_generalist import run

    summary = run(
        Path("configs/experiments/phase1_qwen38_local.yaml"),
        output_path=tmp_path / "qwen38-openai-live-events.jsonl",
        backend_override="openai_compatible",
        model_override=model,
        base_url_override=base_url,
        api_key_env_override=os.environ.get("EVORS_LIVE_OPENAI_API_KEY_ENV"),
    )

    assert summary["status"] == "passed", summary
    assert summary["validation_scope"] == "real_vlm_strict_gate"
    assert summary["scientific_claim"] is False
    assert summary["gate"]["strict"] is True
    assert summary["gate"]["passed"] is True
    assert summary["gate"]["expected_model_id"] == model
    assert summary["sample_count"] == 2
    assert {item["answer"] for item in summary["results"]} == {"3", "6"}
    for item in summary["results"]:
        assert item["completed"] is True
        assert item["gate_passed"] is True
        assert item["gate_failures"] == []
        assert item["model_id"] == model
        assert item["score"]["correct"] is True
        assert item["cost"]["model_input_tokens"] > 0
        assert item["cost"]["model_output_tokens"] > 0
        assert item["cost"]["tool_calls"] >= 1
        assert item["evidence_ids"]
