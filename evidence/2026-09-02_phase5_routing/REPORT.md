# EvoRS Phase 5 Routing Mechanism Gate

- Scope: deterministic control-learning mechanism gate, not a paper benchmark.
- Initial held-out reward: 0.569063
- Learned held-out reward: 0.631716
- Held-out accuracy: 1.000
- Initial mean Agent calls: 4.000
- Learned mean Agent calls: 3.500
- Learned paths: {"visual->knowledge->reasoning->verification": 2, "visual->reasoning->verification": 2}
- Policy parameters changed: yes
- Cognitive Expert fingerprints unchanged: yes
- Every sampled action linked to its pre-state transition: yes
- Evaluation parameter updates: zero
- verl installed in this environment: False

The learned greedy policy skips Knowledge for held-out quantity tasks and keeps
Knowledge for held-out airport classification tasks. References are available
only to the reward environment, never to Controller or Expert observations.
