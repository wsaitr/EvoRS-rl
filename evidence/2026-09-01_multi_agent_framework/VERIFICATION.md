# Phase 4 real multi-Agent verification

Date: 2026-09-01

Model: `evors/qwen3.8-27b-q4`

Sample: `vrsbench-val-001` — windmill counting, reference answer `6`.
The reference answer is used only by the experiment scorer after each episode.

## Results

| Arm | Answer | Correct | Workspace | Messages | Artifacts | Latent bytes |
|---|---:|---|---:|---:|---:|---:|
| text_structure | 6 | yes | v3 | 2 | 7 | 0 |
| qwen_native_latent | 6 | yes | v3 | 2 | 14 | 143,360 |

Both arms executed `visual -> reasoning -> verification`. The Visual Expert
ran active multi-view perception plus private evolution with four evaluated
candidates, three variations, and three completed generations. It committed
Evidence/SpatialNode/Artifact descriptors but no AnswerCandidate. Reasoning
created the proposed AnswerCandidate and Verification created the verified
candidate selected by the EvolutionController.

The latent arm committed seven Qwen-native latent descriptors. The Visual
inner loop loaded 143,360 physical bytes; the routed message charged the same
143,360 latent bytes; Reasoning consumed the authorized latents and reported
13 diagnostics.

## Artifacts

- `text_structure/events.jsonl`: complete real episode and transition trace.
- `text_structure/summary.json`: Text + Structure run summary.
- `qwen_native_latent/events.jsonl`: complete real latent episode and trace.
- `qwen_native_latent/summary.json`: latent run summary.

Scope: mechanism/smoke verification only, not a paper-level accuracy claim.

Full regression: `328 passed, 3 skipped`.
