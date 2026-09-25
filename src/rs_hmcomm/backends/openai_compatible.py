"""OpenAI-compatible chat-completions backend for local multimodal servers.

The implementation intentionally uses only the Python standard library for HTTP.
Images and authorization headers are runtime-only and are never included in raised
errors or the returned structured metadata.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from copy import deepcopy
from io import BytesIO
import json
import math
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image

from rs_hmcomm.backends.base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
    VisualInput,
)


class OpenAICompatibleBackendError(RuntimeError):
    """A secret-safe transport or response-contract failure."""


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("extra_body must contain only finite JSON values") from exc
    if not isinstance(decoded, dict):  # defensive; ``dict(value)`` guarantees this
        raise ValueError("extra_body must be a JSON object")
    return decoded


def _endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty HTTP(S) URL")
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = f"{path}/chat/completions"
    else:
        endpoint_path = f"{path}/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def _non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAICompatibleBackendError(
            f"chat-completions response has invalid {name}"
        )
    return value


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise OpenAICompatibleBackendError(
                    "chat-completions response has invalid message content"
                )
            part_type = part.get("type")
            text = part.get("text")
            if part_type in {"text", "output_text"} and isinstance(text, str):
                chunks.append(text)
        if chunks:
            return "".join(chunks)
    raise OpenAICompatibleBackendError(
        "chat-completions response contains no textual assistant output"
    )


class OpenAICompatibleVLMBackend:
    """Call a local OpenAI-compatible ``/v1/chat/completions`` VLM endpoint."""

    _RESERVED_EXTRA_FIELDS = {"model", "messages", "max_tokens", "stream"}
    _MAX_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:12434/engines",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Mapping[str, Any] | None = None,
        enable_thinking: bool | None = False,
        preserve_thinking: bool | None = False,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        timeout_value = float(timeout)
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise ValueError("timeout must be a positive finite number")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        for name, value in (
            ("enable_thinking", enable_thinking),
            ("preserve_thinking", preserve_thinking),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be bool or None")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be a string or None")

        copied_extra = _json_copy(extra_body or {})
        conflicts = self._RESERVED_EXTRA_FIELDS.intersection(copied_extra)
        if conflicts:
            raise ValueError(
                f"extra_body cannot override reserved fields: {sorted(conflicts)}"
            )
        chat_kwargs = copied_extra.get("chat_template_kwargs")
        if chat_kwargs is not None and not isinstance(chat_kwargs, dict):
            raise ValueError("extra_body.chat_template_kwargs must be an object")

        self.model = model.strip()
        self.endpoint = _endpoint(base_url)
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.timeout = timeout_value
        self.max_tokens = max_tokens
        self.extra_body = copied_extra
        self.enable_thinking = enable_thinking
        self.preserve_thinking = preserve_thinking

    @staticmethod
    def _image_data_uri(image: Any) -> str:
        if not isinstance(image, Image.Image):
            raise OpenAICompatibleBackendError("visual input must be a PIL image")
        buffer = BytesIO()
        try:
            image.save(buffer, format="PNG")
        except Exception:
            raise OpenAICompatibleBackendError("visual input could not be encoded") from None
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def _build_payload(self, request: BackendRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for visual in request.visuals:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_uri(visual.image),
                        "detail": "auto",
                    },
                }
            )
        content.append({"type": "text", "text": request.prompt})

        extra = deepcopy(self.extra_body)
        supplied_chat_kwargs = extra.pop("chat_template_kwargs", {})
        chat_kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = self.enable_thinking
        if self.preserve_thinking is not None:
            chat_kwargs["preserve_thinking"] = self.preserve_thinking
        chat_kwargs.update(supplied_chat_kwargs)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if chat_kwargs:
            payload["chat_template_kwargs"] = chat_kwargs
        payload.update(extra)
        return payload

    def _request(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        body = json.dumps(
            dict(payload), allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "EvoRS-Comm/0.1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
                status = response.getcode()
        except HTTPError as exc:
            try:
                exc.close()
            finally:
                raise OpenAICompatibleBackendError(
                    f"chat-completions request failed with HTTP {exc.code}"
                ) from None
        except (URLError, TimeoutError, OSError):
            raise OpenAICompatibleBackendError(
                "chat-completions request failed at the transport layer"
            ) from None
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        if status is not None and not 200 <= status < 300:
            raise OpenAICompatibleBackendError(
                f"chat-completions request returned HTTP {status}"
            )
        if len(raw) > self._MAX_RESPONSE_BYTES:
            raise OpenAICompatibleBackendError("chat-completions response is too large")
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise OpenAICompatibleBackendError(
                "chat-completions response is not valid JSON"
            ) from None
        if not isinstance(decoded, dict):
            raise OpenAICompatibleBackendError(
                "chat-completions response must be a JSON object"
            )
        return decoded, latency_ms

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        payload = self._build_payload(request)
        response, latency_ms = self._request(payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenAICompatibleBackendError(
                "chat-completions response contains no choice"
            )
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise OpenAICompatibleBackendError(
                "chat-completions choice contains no assistant message"
            )
        text = _message_text(message.get("content"))

        usage_object = response.get("usage")
        if usage_object is None:
            input_tokens = output_tokens = None
        elif isinstance(usage_object, dict):
            input_tokens = _non_negative_int(
                usage_object.get("prompt_tokens", usage_object.get("input_tokens")),
                "input token usage",
            )
            output_tokens = _non_negative_int(
                usage_object.get(
                    "completion_tokens", usage_object.get("output_tokens")
                ),
                "output token usage",
            )
        else:
            raise OpenAICompatibleBackendError(
                "chat-completions response has invalid usage"
            )
        accounting = (
            TokenAccounting.MEASURED
            if input_tokens is not None and output_tokens is not None
            else TokenAccounting.UNAVAILABLE
        )

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise OpenAICompatibleBackendError(
                "chat-completions response has invalid finish_reason"
            )
        response_model = response.get("model")
        if response_model is not None and not isinstance(response_model, str):
            raise OpenAICompatibleBackendError(
                "chat-completions response has invalid model"
            )
        normalized_response_model = (
            response_model.strip() if isinstance(response_model, str) else ""
        )
        revision = response.get("model_revision", response.get("system_fingerprint", ""))
        if revision is None:
            revision = ""
        if not isinstance(revision, str):
            raise OpenAICompatibleBackendError(
                "chat-completions response has invalid model revision"
            )
        response_id = response.get("id")
        safe_structured: dict[str, Any] = {
            "finish_reason": finish_reason,
            # Keep provenance explicit. Falling back to the requested model is
            # convenient for integration, but it must not let a text stub pass
            # the real-model identity gate.
            "response_model_reported": bool(normalized_response_model),
            "response_model_id": response_model,
        }
        if isinstance(response_id, str):
            safe_structured["response_id"] = response_id

        return AgentOutput(
            text=text,
            structured=safe_structured,
            usage=BackendUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                stop_reason=finish_reason,
                token_accounting=accounting,
            ),
            model_id=normalized_response_model or self.model,
            model_revision=revision,
        )

    def generate(
        self,
        image: Any,
        prompt: str,
        structured_context: dict[str, Any] | None = None,
    ) -> AgentOutput:
        return self.generate_turn(
            BackendRequest(
                prompt=prompt,
                visuals=(VisualInput("input_image", "input_image", image),),
                structured_context=structured_context or {},
            )
        )

    def encode_roi(
        self,
        image: Any,
        bbox: tuple[float, float, float, float],
    ) -> None:
        return None


__all__ = ["OpenAICompatibleBackendError", "OpenAICompatibleVLMBackend"]
