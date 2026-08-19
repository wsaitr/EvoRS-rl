# Framework decision

**Stage 1:** own orchestration in this repo.

**Stage 2 first:** CoMLRL/MAGRPO.
It is closer to the paper's cooperative MARL requirement and lighter for algorithm prototyping.

**MARTI second:** use when needing graph workflow runtime, async tools, heterogeneous agents,
distributed large-scale training or tree-search experiments.

Do not assume either upstream repository is Ascend-ready without adaptation.
Keep portability boundary at:
- dataset
- shared state
- message protocol
- trajectory schema
- reward
