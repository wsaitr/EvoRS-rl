# Single-Agent Perception Gate Evidence

Date: 2026-09-01

This directory freezes the first complete paired development gate where the
single active-perception Agent outperformed one-shot use of the same VLM.

## Fixed setup

- Model: `evors/qwen3.8-27b-q4`
- Seed: `20260831`
- Manifest: `configs/experiments/vrsbench20_manifest.jsonl`
- Samples: 20 (10 object-quantity + 10 object-existence)
- Runtime access to reference answers: prohibited; references are used only by
  the offline scorer

## Result

| Arm | Correct | Accuracy | Quantity | Existence | Errors |
|---|---:|---:|---:|---:|---:|
| Direct | 10/20 | 50% | 4/10 | 6/10 | 0 |
| Single Agent | 12/20 | 60% | 6/10 | 6/10 | 0 |

The Agent gained two correct counting answers without regressing any sample
that Direct answered correctly. This passes the development gate
`agent_correct > direct_correct`; it is not a paper-level significance claim.

## Measured cost

| Arm | Model calls | Tool calls | Input tokens | Output tokens | Summed wall time |
|---|---:|---:|---:|---:|---:|
| Direct | 20 | 0 | 6,117 | 41 | 161.07 s |
| Single Agent | 47 | 15 | 25,475 | 1,938 | 871.83 s |

## Files and integrity

- `events.jsonl`: complete paired per-sample records, SHA-256
  `8D3D5C4B06E37274A95659DF94E92E1A2E9BE1DA47B9295D43A56E50B3FE41CF`
- `summary.json`: aggregate plus embedded records, SHA-256
  `FF7EC7709F833B077E29500BE286DF866F3B8FEB8BDDE0645EEFFAB29A3BBF28`
- Manifest SHA-256:
  `1ba99639595b57e04e3cd3bda6714553c2ba946197640cfcacb0512ca705c046`
- Model-config SHA-256:
  `ee30ae9b41d4f617c2554a967649554a1ebd17e35882ea3740806e7957a7fe3b`

## Verification

```powershell
$env:PYTHONPATH = "src"
python scripts/run_single_agent_perception_gate.py `
  --output-dir evidence/2026-09-01_single_agent_perception_gate `
  --overwrite
python -m pytest -q
```

The repository-wide offline test result for this evidence point was
`319 passed, 3 skipped in 82.80s`.
