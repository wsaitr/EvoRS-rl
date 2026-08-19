from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    device: str
    distributed_backend: str

def resolve_device(preference: str = "auto") -> DeviceSpec:
    preference = (preference or "auto").lower()

    if preference in {"npu", "ascend"}:
        try:
            import torch_npu  # noqa: F401
            return DeviceSpec("npu", "npu:0", "hccl")
        except Exception as e:
            raise RuntimeError("Ascend requested but torch_npu is unavailable") from e

    if preference == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return DeviceSpec("cuda", "cuda:0", "nccl")

    if preference == "cpu":
        return DeviceSpec("cpu", "cpu", "gloo")

    try:
        import torch_npu  # noqa: F401
        if os.environ.get("ASCEND_VISIBLE_DEVICES") or os.environ.get("NPU_VISIBLE_DEVICES"):
            return DeviceSpec("npu", "npu:0", "hccl")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return DeviceSpec("cuda", "cuda:0", "nccl")
    except Exception:
        pass

    try:
        import torch_npu  # noqa: F401
        return DeviceSpec("npu", "npu:0", "hccl")
    except Exception:
        return DeviceSpec("cpu", "cpu", "gloo")
