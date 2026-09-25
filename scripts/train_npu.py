#!/usr/bin/env python3
"""Ascend/lws2 entry for the Phase 5 routing controller.

The current controller is a tiny discrete policy and intentionally trains on
CPU, even when launched on an Ascend host.  This preserves expensive NPU time
for later token-policy or Phase 6 VLM experiments.  ``verify`` records the NPU
and verl environment honestly; it does not claim that a dry-run trained them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rs_hmcomm.control import LearnedRoutingPolicy  # noqa: E402
from rs_hmcomm.rl import verl_environment_status  # noqa: E402


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "evors.ascend-environment.v1",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES"),
        "npu": {"torch_installed": False, "torch_npu_installed": False, "available": False},
        "verl": verl_environment_status(),
    }
    try:
        import torch

        report["npu"]["torch_installed"] = True
        report["npu"]["torch_version"] = torch.__version__
        try:
            import torch_npu  # noqa: F401

            report["npu"]["torch_npu_installed"] = True
            report["npu"]["available"] = bool(torch.npu.is_available())
            report["npu"]["device_count"] = int(torch.npu.device_count())
        except Exception as exc:  # noqa: BLE001 - diagnostic
            report["npu"]["error_type"] = type(exc).__name__
    except Exception as exc:  # noqa: BLE001 - diagnostic
        report["npu"]["error_type"] = type(exc).__name__
    return report


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def verify(
    *,
    checkpoint: Path | None,
    output: Path,
    require_npu: bool,
    require_verl: bool,
) -> dict[str, Any]:
    report = environment_report()
    report["checkpoint"] = None
    if checkpoint is not None:
        policy = LearnedRoutingPolicy.from_checkpoint(checkpoint)
        report["checkpoint"] = {
            "path": str(checkpoint.resolve()),
            "parameter_sha256": policy.parameter_digest(),
            "trained_steps": policy.trained_steps,
        }
    checks = {
        "checkpoint_valid": checkpoint is None or report["checkpoint"] is not None,
        "npu_requirement_satisfied": not require_npu or bool(report["npu"]["available"]),
        "verl_requirement_satisfied": not require_verl or bool(report["verl"]["installed"]),
    }
    report["observed"] = {
        "npu_available": bool(report["npu"]["available"]),
        "verl_installed": bool(report["verl"]["installed"]),
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    _write(output, report)
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Ascend environment verification failed: {failed}")
    return report


def train_routing(config: Path, output: Path) -> dict[str, Any]:
    from train_phase5_routing import run

    # Explicitly CPU: the trainable controller is only a 7x21 linear policy.
    result = run(config.resolve(), output_override=output.resolve())
    result["execution_device"] = "cpu-controller"
    result["npu_reserved_for_vlm"] = True
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("verify", "routing"),
        help="verify the host, or execute real Phase 5 controller training",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "phase5_routing.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "evidence" / "2026-09-03_phase5_routing" / "checkpoint.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-npu", action="store_true")
    parser.add_argument("--require-verl", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "verify":
        result = verify(
            checkpoint=args.checkpoint,
            output=args.output,
            require_npu=args.require_npu,
            require_verl=args.require_verl,
        )
    else:
        result = train_routing(args.config, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
