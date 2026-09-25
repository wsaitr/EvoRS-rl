#!/usr/bin/env python3
"""Prepare an offline CPython 3.10/aarch64 dependency tree for ModelArts.

The target Ascend image already owns the CANN-coupled torch and torch_npu
installation.  This script resolves the remaining project dependencies as ARM64
wheels, excludes accelerator-framework wheels, and expands them into a vendor
tree that can be copied by Docker without executing ARM code on the x86 build
host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "docker" / "requirements-modelarts.txt"
DEFAULT_WHEELHOUSE = ROOT / ".codex_work" / "modelarts-wheelhouse"
DEFAULT_VENDOR = ROOT / ".modelarts" / "vendor"

# These must come from the Ascend/CANN base image.  Vendoring a PyPI build would
# silently replace torch_npu-compatible binaries.
EXCLUDED_DISTRIBUTIONS = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytorch-triton",
}


def _distribution_name(wheel: Path) -> str:
    return wheel.name.split("-", 1)[0].replace("_", "-").lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe wheel member: {name}")
    return path


def _target_for_member(member: str) -> Path | None:
    path = _safe_member_path(member)
    parts = path.parts
    if not parts:
        return None
    if ".data" in parts[0]:
        if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
            return None
        return Path(*parts[2:])
    return path


def _extract_wheel(wheel: Path, destination: Path) -> int:
    written = 0
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            target_relative = _target_for_member(info.filename)
            if target_relative is None or info.is_dir():
                continue
            target = destination / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            written += 1
    return written


def _download(requirements: Path, wheelhouse: Path) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--requirement",
        str(requirements),
        "--dest",
        str(wheelhouse),
        "--only-binary=:all:",
        "--platform=manylinux2014_aarch64",
        "--implementation=cp",
        "--python-version=310",
        "--abi=cp310",
        "--progress-bar=off",
        "--disable-pip-version-check",
    ]
    subprocess.run(command, check=True)


def prepare(requirements: Path, wheelhouse: Path, vendor: Path) -> dict:
    _download(requirements, wheelhouse)
    if vendor.exists():
        shutil.rmtree(vendor)
    vendor.mkdir(parents=True)

    included: list[dict] = []
    excluded: list[str] = []
    files_written = 0
    wheels: Iterable[Path] = sorted(wheelhouse.glob("*.whl"))
    for wheel in wheels:
        distribution = _distribution_name(wheel)
        if distribution in EXCLUDED_DISTRIBUTIONS:
            excluded.append(wheel.name)
            continue
        files_written += _extract_wheel(wheel, vendor)
        included.append(
            {
                "file": wheel.name,
                "sha256": _sha256(wheel),
                "bytes": wheel.stat().st_size,
            }
        )

    if (vendor / "torch" / "__init__.py").exists():
        raise RuntimeError("vendor tree must not contain torch")
    if any(vendor.rglob("*.pyd")):
        raise RuntimeError("Windows extension module found in ARM64 vendor tree")

    manifest = {
        "schema_version": "evors.modelarts-vendor.v1",
        "platform": "manylinux2014_aarch64",
        "python": "cp310",
        "requirements": str(requirements.relative_to(ROOT)),
        "included_wheels": included,
        "excluded_accelerator_wheels": excluded,
        "files_written": files_written,
    }
    (vendor / "VENDOR-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--wheelhouse", type=Path, default=DEFAULT_WHEELHOUSE)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    args = parser.parse_args()
    result = prepare(
        args.requirements.resolve(), args.wheelhouse.resolve(), args.vendor.resolve()
    )
    print(
        json.dumps(
            {
                "included_wheels": len(result["included_wheels"]),
                "excluded_accelerator_wheels": result[
                    "excluded_accelerator_wheels"
                ],
                "files_written": result["files_written"],
                "vendor": str(args.vendor.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
