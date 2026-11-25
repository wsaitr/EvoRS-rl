#!/usr/bin/env python3
"""One-click inference over every agent benchmark supported by run_agent_benchmarks.py.

This only orchestrates the existing benchmark entry; it does not change prompts,
decoding, scoring, datasets, or model code.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

BENCHMARKS = [
    ("GeoMMBench", "all"),
    ("ThinkGeo", "optical"),
    ("Earth-Bench", "rgb"),
    ("GeoPlan", "all"),
]
METHODS = ("pure_model", "single_agent", "text_multi", "innovation_multi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--data", required=True, help="benchmark data root")
    parser.add_argument("--output-root", required=True, help="new directory for all runs")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="comma-separated subset of " + ",".join(METHODS))
    parser.add_argument("--split", default="dev", choices=("dev", "train", "test"))
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = full split; otherwise first N examples per benchmark")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    entry = root / "scripts" / "run_agent_benchmarks.py"
    if not entry.is_file():
        parser.error(f"benchmark entry not found: {entry}")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown methods: {unknown}")

    output_root = Path(args.output_root).resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=False)

    for benchmark, modality in BENCHMARKS:
        for method in methods:
            output = output_root / f"{benchmark}_{method}"
            cmd = [
                sys.executable, str(entry),
                "--model", str(Path(args.model).resolve()),
                "--data", str(Path(args.data).resolve()),
                "--benchmark", benchmark,
                "--method", method,
                "--split", args.split,
                "--modality", modality,
                "--output", str(output),
            ]
            if args.limit > 0:
                cmd += ["--limit", str(args.limit)]
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
