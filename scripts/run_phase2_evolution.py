"""Run the Phase 2 single-Agent private evolution mechanism."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from scripts.run_phase1_generalist import run
except ModuleNotFoundError:  # Direct execution: python scripts/run_phase2_evolution.py
    from run_phase1_generalist import run


DEFAULT_CONFIG = Path("configs/experiments/phase2_evolution.yaml")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        args.config,
        output_path=args.output,
        overwrite=args.overwrite,
        expected_phase=2,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
