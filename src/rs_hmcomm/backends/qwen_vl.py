"""Model-specific Qwen2.5-VL backend for the Phase 1 live gate.

This follows the official Qwen2.5-VL Transformers + ``qwen_vl_utils`` chat
path.  Importing this module is light; weights are loaded only on construction.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .base import (
    AgentOutput,
    BackendRequest,
    BackendUsage,
    TokenAccounting,
    VisualInput,
)
from .device import resolve_device


class Qwen25VLBackend:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 256,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        local_files_only: bool = True,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if min_pixels is not None and min_pixels <= 0:
            raise ValueError("min_pixels must be positive")
        if max_pixels is not None and max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        if min_pixels is not None and max_pixels is not None and min_pixels > max_pixels:
            raise ValueError("min_pixels cannot exceed max_pixels")

        self.model_name = model_name_or_path
        self.requested_revision = revision
        self.spec = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen25VLBackend requires a Transformers release with Qwen2.5-VL support"
            ) from exc
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("Qwen25VLBackend requires qwen-vl-utils") from exc

        load_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "local_files_only": local_files_only,
        }
        processor_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            load_kwargs["revision"] = revision
            processor_kwargs["revision"] = revision
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        self.model.to(self.spec.device)
        self.model.eval()
        self._process_vision_info = process_vision_info
        self.model_revision = (
            revision
            or getattr(self.model.config, "_commit_hash", None)
            or "local-unpinned"
        )

    def _build_messages(self, request: BackendRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for visual in request.visuals:
            image_item: dict[str, Any] = {
                "type": "image",
                "image": visual.image,
            }
            if self.min_pixels is not None:
                image_item["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                image_item["max_pixels"] = self.max_pixels
            content.append(image_item)
        content.append({"type": "text", "text": request.prompt})
        return [{"role": "user", "content": content}]

    def generate_turn(self, request: BackendRequest) -> AgentOutput:
        import torch

        messages = self._build_messages(request)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.spec.device)
        input_tokens = int(inputs["input_ids"].numel())

        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        prompt_length = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_length:]
        output_tokens = int(trimmed.numel())
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        stop_reason = (
            "max_new_tokens" if output_tokens >= self.max_new_tokens else "eos_or_stop_token"
        )
        return AgentOutput(
            text=text,
            usage=BackendUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                stop_reason=stop_reason,
                token_accounting=TokenAccounting.MEASURED,
            ),
            model_id=self.model_name,
            model_revision=self.model_revision,
        )

    def generate(
        self,
        image: Any,
        prompt: str,
        structured_context: Optional[dict[str, Any]] = None,
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
        # Phase 3 owns explicit latent extraction and its descriptor contract.
        return None


__all__ = ["Qwen25VLBackend"]
