# Integrated EvoRS Single-Agent Evidence

Date: 2026-09-01

This evidence freezes the first real episode connecting the previously
separate Phase 1-3 mechanisms:

```text
active perception
  -> perception WorkspaceDelta
  -> question-conditioned Scene/Group/Object hierarchy
  -> ArtifactStore descriptors
  -> scoped Workspace read
  -> optional learned latent consumption
  -> private (1+1) candidate evolution
  -> selected Claim/AnswerCandidate WorkspaceDelta
```

No Controller, peer Agent, Router, RL policy, reference answer, or GT bbox is
available to the runtime.

## Real Qwen smoke

- Model: `evors/qwen3.8-27b-q4`
- Seed: `20260831`
- Sample: `vrsbench-val-001`
- Reference: 6
- Integrated Agent answer: 6
- Perception rule: `enhanced_count_localization_consensus`
- Model/tool calls: 4 / 1
- Evolution: 3 generations, 4 candidates evaluated, 3 variations
- Public Workspace: version 2, one selected AnswerCandidate
- Hierarchy: Scene -> Group -> 6 Object nodes

### Text + Structure

- Workspace artifact descriptors: 7
- Latent bytes loaded: 0
- Wall time: 108.76 s
- Files: `text_structure/events.jsonl`, `text_structure/summary.json`

### Qwen native learned latent

- Encoder: `qwen_native_v1`
- Latent descriptors: 1 Scene + 6 ROI
- Workspace artifact descriptors: 14
- Values loaded: 35,840 float32
- Bytes loaded: 143,360
- ROI mean cosine similarity: 0.984916387877485
- Latent fitness component differed across candidates: yes
- Wall time: 158.45 s
- Files: `qwen_native_latent/events.jsonl`,
  `qwen_native_latent/summary.json`

The Workspace JSON contains descriptors and stable references only. Image,
crop, and latent payloads remain in the ArtifactStore.

## Integrity

- `text_structure/events.jsonl`:
  `E29C63BA8D87E2C38139A57FF8F4A4722BA10D703336BF03B0D8F27CA0A23446`
- `text_structure/summary.json`:
  `6E95D28F36E8E631837B90965DE982CA906C780C67016B5AC3C87D745F925DE5`
- `qwen_native_latent/events.jsonl`:
  `ADAC1378BEC8738F93A8474BF179FE5C447B931E14775E1121D26DF501CDB0DB`
- `qwen_native_latent/summary.json`:
  `F87B5305B8F8C47F94246031F1A0D51116F3139A5836CB40FFE2485F97634E27`

## Verification

```powershell
$env:PYTHONPATH = "src"
python scripts/run_complete_single_agent.py `
  --output-dir evidence/2026-09-01_integrated_single_agent/qwen_native_latent `
  --offset 1 --limit 1 --latent qwen_native --overwrite
python -m pytest -q
```

Repository-wide test result: `323 passed, 3 skipped in 61.14s`.

This is a mechanism smoke, not a paper-level latent effectiveness result.
