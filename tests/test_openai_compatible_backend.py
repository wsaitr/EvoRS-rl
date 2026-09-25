from __future__ import annotations

import base64
from io import BytesIO
import json
from urllib.error import HTTPError, URLError

from PIL import Image
import pytest

from rs_hmcomm.backends.base import BackendRequest, TokenAccounting, VisualInput
from rs_hmcomm.backends.openai_compatible import (
    OpenAICompatibleBackendError,
    OpenAICompatibleVLMBackend,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.raw if size < 0 else self.raw[:size]

    def getcode(self) -> int:
        return self.status


def _request() -> BackendRequest:
    return BackendRequest(
        prompt="Return one JSON action.",
        visuals=(
            VisualInput("input_image", "full image", Image.new("RGB", (4, 3), "red")),
            VisualInput(
                "crop-1",
                "crop",
                Image.new("RGB", (2, 2), "blue"),
                (0.0, 0.0, 0.5, 0.5),
            ),
        ),
    )


def test_chat_completions_payload_and_usage_mapping(monkeypatch) -> None:
    captured: dict[str, object] = {}
    model_text = '{"type":"finish","answer":"6","evidence_ids":["ev"],"confidence":0.9}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "id": "chatcmpl-local-1",
                "model": "qwen3.8:27b-q4",
                "system_fingerprint": "local-revision",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": model_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": 27,
                    "total_tokens": 348,
                },
            }
        )

    monkeypatch.setattr(
        "rs_hmcomm.backends.openai_compatible.urlopen", fake_urlopen
    )
    backend = OpenAICompatibleVLMBackend(
        "qwen3.8:27b-q4",
        base_url="http://127.0.0.1:12434/engines/v1",
        api_key="local-secret",
        timeout=7.5,
        max_tokens=192,
        extra_body={
            "temperature": 0,
            "chat_template_kwargs": {"custom_flag": "value"},
        },
    )

    output = backend.generate_turn(_request())

    assert captured["url"] == "http://127.0.0.1:12434/engines/v1/chat/completions"
    assert captured["timeout"] == 7.5
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer local-secret"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen3.8:27b-q4"
    assert payload["max_tokens"] == 192
    assert payload["stream"] is False
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
        "custom_flag": "value",
    }
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[-1]["text"] == "Return one JSON action."
    first_uri = content[0]["image_url"]["url"]
    assert first_uri.startswith("data:image/png;base64,")
    decoded_image = Image.open(BytesIO(base64.b64decode(first_uri.split(",", 1)[1])))
    assert decoded_image.size == (4, 3)

    assert output.text == model_text
    assert output.model_id == "qwen3.8:27b-q4"
    assert output.model_revision == "local-revision"
    assert output.structured == {
        "finish_reason": "stop",
        "response_model_reported": True,
        "response_model_id": "qwen3.8:27b-q4",
        "response_id": "chatcmpl-local-1",
    }
    assert output.usage.input_tokens == 321
    assert output.usage.output_tokens == 27
    assert output.usage.stop_reason == "stop"
    assert output.usage.token_accounting == TokenAccounting.MEASURED
    assert output.usage.latency_ms >= 0


@pytest.mark.parametrize("failure", ["http", "transport"])
def test_transport_failures_do_not_expose_key_image_or_server_body(
    monkeypatch, failure: str
) -> None:
    secret = "api-key-must-not-leak"
    response_secret = "server-echoed-private-payload"

    def fake_urlopen(request, timeout):
        if failure == "http":
            raise HTTPError(
                request.full_url,
                500,
                response_secret,
                None,
                BytesIO(response_secret.encode("utf-8")),
            )
        raise URLError(f"network failure {secret} {response_secret}")

    monkeypatch.setattr(
        "rs_hmcomm.backends.openai_compatible.urlopen", fake_urlopen
    )
    backend = OpenAICompatibleVLMBackend(
        "local-model",
        base_url="http://127.0.0.1:9000",
        api_key=secret,
    )

    with pytest.raises(OpenAICompatibleBackendError) as caught:
        backend.generate_turn(_request())

    message = str(caught.value)
    assert secret not in message
    assert response_secret not in message
    assert "base64" not in message


def test_missing_usage_remains_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "rs_hmcomm.backends.openai_compatible.urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": [{"type": "text", "text": "{}"}]},
                        "finish_reason": "length",
                    }
                ]
            }
        ),
    )
    output = OpenAICompatibleVLMBackend(
        "local-model", base_url="http://127.0.0.1:9000/v1"
    ).generate_turn(_request())

    assert output.text == "{}"
    assert output.usage.input_tokens is None
    assert output.usage.output_tokens is None
    assert output.usage.token_accounting == TokenAccounting.UNAVAILABLE
    assert output.structured["response_model_reported"] is False
    assert output.structured["response_model_id"] is None


@pytest.mark.parametrize("reported_model", ["", "   "])
def test_empty_server_model_cannot_claim_explicit_identity(
    monkeypatch, reported_model: str
) -> None:
    monkeypatch.setattr(
        "rs_hmcomm.backends.openai_compatible.urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "model": reported_model,
                "choices": [
                    {
                        "message": {"content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ),
    )

    output = OpenAICompatibleVLMBackend(
        "requested-model", base_url="http://127.0.0.1:9000/v1"
    ).generate_turn(_request())

    assert output.model_id == "requested-model"
    assert output.structured["response_model_reported"] is False
    assert output.structured["response_model_id"] == reported_model


def test_configuration_rejects_secret_in_url_and_reserved_overrides() -> None:
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleVLMBackend(
            "local-model", base_url="http://user:password@127.0.0.1:9000"
        )
    with pytest.raises(ValueError, match="reserved"):
        OpenAICompatibleVLMBackend(
            "local-model",
            base_url="http://127.0.0.1:9000",
            extra_body={"messages": []},
        )


def test_phase1_backend_factory_reads_api_key_only_from_named_environment(
    monkeypatch,
) -> None:
    from scripts.run_phase1_generalist import build_backend

    monkeypatch.setenv("EVORS_TEST_LOCAL_KEY", "test-key")
    backend, scope = build_backend(
        {
            "type": "openai_compatible",
            "model": "local-qwen",
            "base_url": "http://127.0.0.1:9000",
            "api_key_env": "EVORS_TEST_LOCAL_KEY",
            "max_tokens": 64,
            "extra_body": {},
        },
        backend_override=None,
        model_path_override=None,
        device_override=None,
        allow_download=False,
    )

    assert isinstance(backend, OpenAICompatibleVLMBackend)
    assert scope == "real_vlm_strict_gate"
    assert backend.model == "local-qwen"
    assert backend.max_tokens == 64
