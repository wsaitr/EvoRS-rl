# EvoRS Phase 4 paired comparison

Status: `completed`

Design: paired frozen perception. Reference answers were used only by the scorer.

## Accuracy and mean cost

| Arm | Correct | Accuracy (Wilson 95% CI) | Model calls | Tool calls | Wall time (ms) | Text tokens | Structure bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_vlm | 10/20 | 0.500 [0.299, 0.701] | 1.00 | 0.00 | 8053.4 | 0.0 | 0.0 |
| single_agent | 12/20 | 0.600 [0.387, 0.781] | 2.35 | 0.75 | 43625.2 | 0.0 | 0.0 |
| multi_agent | 14/20 | 0.700 [0.481, 0.855] | 4.35 | 0.75 | 65945.7 | 260.8 | 771.4 |

## Paired outcomes

- `single_vs_direct`: wins 2, losses 0, ties 18 over 20 samples; exact McNemar p=0.500.
- `multi_vs_direct`: wins 4, losses 0, ties 16 over 20 samples; exact McNemar p=0.125.
- `multi_vs_single`: wins 2, losses 0, ties 18 over 20 samples; exact McNemar p=0.500.

## Task-type accuracy

| Task type | Direct | Single | Multi |
|---|---:|---:|---:|
| object_existence | 6/10 | 6/10 | 8/10 |
| object_quantity | 4/10 | 6/10 | 6/10 |

## Changed Multi-Agent answers

- `vrsbench-val-004`: Single `3` -> Multi `4`; reference `2`.
- `vrsbench-val-015`: Single `Yes` -> Multi `No`; reference `no`.
- `vrsbench-val-016`: Single `Yes` -> Multi `No`; reference `no`.

## Integrity

- `unique_sample_ids`: `true`
- `all_arms_error_free`: `true`
- `multi_agent_paths_valid`: `true`
- `multi_agent_task_label_isolation`: `true`
- `multi_agent_replay_material_present`: `true`

## Scope

This is a 20-sample development comparison, not a paper-level statistical conclusion. Direct and Visual perception outputs are frozen from the cited baseline evidence; Multi Agent adds newly executed Reasoning and Verification calls.
