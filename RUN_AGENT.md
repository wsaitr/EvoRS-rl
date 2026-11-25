# EvoRS Agent: minimal inference / benchmark entry

This directory contains only the Agent core and the two launch paths requested:
one-shot Agent inference, and one-click inference over every supported agent
benchmark. Development tests, paper scripts, experiment history and outputs are
intentionally not included.

## Environment

```bash
python -m pip install -e ".[dev]"
# or reuse the project's existing Python environment
export PYTHONPATH="$PWD/src:$PWD"
```

The default local model config is `configs/models/qwen38_27b_q4_local.yaml`.
Edit its model path for the new machine. Model weights and benchmark data are not
stored in this repository; point the commands below at their local paths.

## 1) One-shot Agent inference

```bash
python scripts/run_real_multi_agent.py \
  --output-dir runs/infer_demo \
  --limit 1
```

Defaults:
- manifest: `configs/experiments/vrsbench20_manifest.jsonl`
- model config: `configs/models/qwen38_27b_q4_local.yaml`

For the full VRSBench real-pipeline comparison (direct / flat / hierarchical /
learned routing), use `scripts/run_real_pipeline.py --manifest ... --output-dir ...`.

## 2) One-click all agent benchmark sets

```bash
python scripts/run_all_benchmarks.py \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --data /path/to/agent-benchmarks \
  --output-root runs/agentbench_all \
  --methods pure_model,single_agent,text_multi,innovation_multi \
  --split dev
```

Supported benchmark sets and modalities:
- `GeoMMBench` / `all`
- `ThinkGeo` / `optical`
- `Earth-Bench` / `rgb`
- `GeoPlan` / `all`

`--limit 0` runs the full split. Each run is isolated under
`<output-root>/<benchmark>_<method>/`. The script only calls the existing
`scripts/run_agent_benchmarks.py`; prompts, decoding, scoring and data loading
are unchanged.

## 3) Paper-final plan (MME-RealWorld-RS / VLRS / XLRS / LRS-VQA)

If you already have a frozen final plan:

```bash
python scripts/run_paper_final.py \
  --plan /path/to/PLAN.json \
  --worker 0 \
  --output runs/paper_final_worker0
```

The plan and manifests are data-side artifacts and are not in this repository;
they must be produced by the original data-preparation pipeline.

## Boundaries

- No API keys, tokens or internal hostnames are required by these entry points.
- No test/development scripts, paper figure code, `ops/kml` history, outputs,
  model weights or third-party trees are included.
- `tests/`, `doc/`, `output/`, `components/`, `third_party/` are excluded.
