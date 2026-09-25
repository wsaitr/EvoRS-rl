from __future__ import annotations

import json
from pathlib import Path

from rs_hmcomm.backends.base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
)
from rs_hmcomm.datasets.manifest import iter_manifest_dataset
from rs_hmcomm.runtime.single_agent import SingleAgentRunResult


CONFIG = Path("configs/experiments/phase1_qwen38_local.yaml")
MANIFEST = Path("tests/fixtures/phase1/manifest.jsonl")
EXPECTED_MODEL = "evors/qwen3.8-27b-q4"


def _count_white_components(image) -> int:
    width, height = image.size
    pixels = image.convert("RGB").load()
    white = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if min(pixels[x, y]) >= 240
    }
    components = 0
    while white:
        stack = [white.pop()]
        size = 0
        while stack:
            x, y = stack.pop()
            size += 1
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in white:
                    white.remove(neighbor)
                    stack.append(neighbor)
        if size >= 100:
            components += 1
    return components


class _VisualFixtureBackend:
    def __init__(
        self,
        *,
        model_id: str = EXPECTED_MODEL,
        response_model_reported: bool = True,
        token_accounting: TokenAccounting = TokenAccounting.MEASURED,
        fixed_answer: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.response_model_reported = response_model_reported
        self.token_accounting = token_accounting
        self.fixed_answer = fixed_answer
        self._next_call = 0
        self._active_call_id = ""

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        if len(request.visuals) == 1:
            self._next_call += 1
            self._active_call_id = f"counterfactual-crop-{self._next_call}"
            text = (
                '{"type":"call_tool","call_id":"'
                + self._active_call_id
                + '","tool_name":"crop","arguments":'
                '{"image_ref":"input_image","bbox":[0,0,1,1]}}'
            )
        else:
            answer = self.fixed_answer or str(
                _count_white_components(request.visuals[0].image)
            )
            text = (
                '{"type":"finish","answer":"'
                + answer
                + '","evidence_ids":["evidence_'
                + self._active_call_id
                + '"]}'
            )
        return AgentOutput(
            text=text,
            structured={
                "response_model_reported": self.response_model_reported,
                "response_model_id": (
                    self.model_id if self.response_model_reported else None
                ),
            },
            usage=BackendUsage(
                input_tokens=128,
                output_tokens=24,
                latency_ms=1.0,
                stop_reason="stop",
                token_accounting=self.token_accounting,
            ),
            model_id=self.model_id,
            model_revision="fixture-revision",
        )


def _run_gate(monkeypatch, tmp_path, backend: _VisualFixtureBackend):
    from scripts import run_phase1_generalist as runner

    monkeypatch.setattr(
        runner,
        "build_backend",
        lambda *args, **kwargs: (backend, "real_vlm_strict_gate"),
    )
    return runner.run(CONFIG, output_path=tmp_path / "events.jsonl")


def test_counterfactual_fixtures_require_image_dependent_answers() -> None:
    samples = tuple(iter_manifest_dataset(MANIFEST))

    assert len(samples) == 2
    assert len({sample.question for sample in samples}) == 1
    assert {sample.answer for sample in samples} == {"3", "6"}
    assert {
        sample.metadata["counterfactual_group"] for sample in samples
    } == {"aircraft-count-v1"}
    assert {
        _count_white_components(sample.image) for sample in samples
    } == {3, 6}


def test_strict_live_gate_accepts_visual_causal_measured_run(
    monkeypatch, tmp_path
) -> None:
    summary = _run_gate(monkeypatch, tmp_path, _VisualFixtureBackend())

    assert summary["status"] == "passed"
    assert summary["validation_scope"] == "real_vlm_strict_gate"
    assert summary["scientific_claim"] is False
    assert summary["gate"]["strict"] is True
    assert summary["gate"]["passed"] is True
    assert [item["answer"] for item in summary["results"]] == ["3", "6"]
    assert all(item["gate_passed"] for item in summary["results"])
    assert all(not item["gate_failures"] for item in summary["results"])


def test_text_stub_that_ignores_images_exits_nonzero(
    monkeypatch, tmp_path, capsys
) -> None:
    from scripts import run_phase1_generalist as runner

    backend = _VisualFixtureBackend(fixed_answer="3")
    monkeypatch.setattr(
        runner,
        "build_backend",
        lambda *args, **kwargs: (backend, "real_vlm_strict_gate"),
    )
    exit_code = runner.main(
        [
            "--config",
            str(CONFIG),
            "--output",
            str(tmp_path / "text-stub-events.jsonl"),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert summary["status"] == "failed"
    assert summary["gate"]["passed"] is False
    assert summary["results"][0]["gate_passed"] is True
    assert summary["results"][1]["gate_passed"] is False
    assert any(
        "visual reference" in failure
        for failure in summary["results"][1]["gate_failures"]
    )


def test_strict_live_gate_rejects_model_fallback_and_unmeasured_usage(
    monkeypatch, tmp_path
) -> None:
    cases = (
        (
            _VisualFixtureBackend(model_id="different-model"),
            "model_id mismatch",
        ),
        (
            _VisualFixtureBackend(response_model_reported=False),
            "explicit server model_id",
        ),
        (
            _VisualFixtureBackend(token_accounting=TokenAccounting.UNAVAILABLE),
            "non-measured token usage",
        ),
    )
    for index, (backend, expected_failure) in enumerate(cases):
        from scripts import run_phase1_generalist as runner

        monkeypatch.setattr(
            runner,
            "build_backend",
            lambda *args, _backend=backend, **kwargs: (
                _backend,
                "real_vlm_strict_gate",
            ),
        )
        summary = runner.run(
            CONFIG,
            output_path=tmp_path / f"events-{index}.jsonl",
        )
        assert summary["status"] == "failed"
        assert any(
            expected_failure in failure
            for result in summary["results"]
            for failure in result["gate_failures"]
        )


def test_runner_invokes_runtime_phase1_causal_validator(monkeypatch, tmp_path) -> None:
    def reject_forged_chain(self, *, require_tool: bool) -> None:
        raise ValueError("forged causal chain")

    monkeypatch.setattr(
        SingleAgentRunResult,
        "validate_phase1_completion",
        reject_forged_chain,
    )
    summary = _run_gate(monkeypatch, tmp_path, _VisualFixtureBackend())

    assert summary["status"] == "failed"
    assert any(
        "forged causal chain" in failure
        for result in summary["results"]
        for failure in result["gate_failures"]
    )
