from __future__ import annotations
from typing import Any, Optional
from .base import AgentOutput
from .device import resolve_device

class TransformersVLMBackend:
    # Minimal direct-Transformers path. Model-specific latent extraction is intentionally separate.
    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 256):
        self.model_name = model_name
        self.spec = resolve_device(device)
        self.max_new_tokens = max_new_tokens

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except Exception:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        self.model = model_cls.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.spec.device)
        self.model.eval()

    def generate(self, image: Any, prompt: str, structured_context: Optional[dict] = None) -> AgentOutput:
        import torch
        context = ""
        if structured_context:
            context = "\nStructured context:\n" + str(structured_context)
        inputs = self.processor(images=image, text=prompt + context, return_tensors="pt")
        moved = {}
        for k, v in inputs.items():
            try:
                moved[k] = v.to(self.spec.device)
            except AttributeError:
                moved[k] = v

        with torch.no_grad():
            out = self.model.generate(**moved, max_new_tokens=self.max_new_tokens)

        if "input_ids" in moved:
            out = out[:, moved["input_ids"].shape[1]:]
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return AgentOutput(text=text)

    def encode_roi(self, image: Any, bbox: tuple[float, float, float, float]) -> Any | None:
        return None
