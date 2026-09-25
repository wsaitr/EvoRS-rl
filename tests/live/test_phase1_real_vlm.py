from __future__ import annotations

import os

import pytest

from rs_hmcomm.agents.generalist import GeneralistAgent
from rs_hmcomm.backends.qwen_vl import Qwen25VLBackend
from rs_hmcomm.contracts import BudgetLimits, StopReason
from rs_hmcomm.datasets.manifest import iter_manifest_dataset
from rs_hmcomm.runtime import SingleAgentRuntime
from rs_hmcomm.tools import CropTool, ToolRegistry


@pytest.mark.live_vlm
def test_real_qwen_generalist_completes_tool_evidence_answer_loop() -> None:
    model_path = os.environ.get("EVORS_LIVE_VLM_MODEL")
    manifest_path = os.environ.get("EVORS_LIVE_RS_MANIFEST")
    if not model_path or not manifest_path:
        pytest.skip("set EVORS_LIVE_VLM_MODEL and EVORS_LIVE_RS_MANIFEST")
    sample = next(iter_manifest_dataset(manifest_path, limit=1))
    backend = Qwen25VLBackend(
        model_path,
        device=os.environ.get("EVORS_LIVE_DEVICE", "auto"),
        local_files_only=True,
        max_new_tokens=256,
        max_pixels=512 * 28 * 28,
    )
    runtime = SingleAgentRuntime(
        GeneralistAgent(backend),
        ToolRegistry((CropTool(),)),
        BudgetLimits(
            max_agent_calls=1,
            max_tool_calls=2,
            max_model_calls=4,
            max_model_input_tokens=65536,
            max_model_output_tokens=1024,
            max_billed_compute_ms=600000,
            max_wall_time_ms=600000,
            max_errors=2,
            max_retries=1,
            max_invalid_actions=2,
        ),
        max_steps=4,
    )
    result = runtime.run(
        sample.image,
        sample.question,
        sample_id=sample.sample_id,
        task_type=sample.task_type,
        require_tool=True,
    )

    assert result.stop_reason == StopReason.COMPLETED
    assert result.answer and result.answer.strip()
    assert result.cost.tool_calls >= 1
    assert result.cost.model_calls >= 2
    assert result.agent_result.evidence
    model_events = [
        event
        for event in result.trace.events
        if event.event_type.value == "model_called"
    ]
    assert all(event.result["token_accounting"] == "measured" for event in model_events)
