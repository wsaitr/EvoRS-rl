# EvoRS Phase 5 Routing Mechanism Gate

- Scope: deterministic control-learning mechanism gate, not a paper benchmark.
- Initial held-out reward: 0.235730
- Learned held-out reward: 0.569049
- Held-out accuracy: 1.000
- Initial mean Agent calls: 4.000
- Learned mean Agent calls: 3.333
- Learned paths: {"visual->knowledge->reasoning->verification": 2, "visual->reasoning->verification": 4}
- Learned actions by task: {"airport_classification": ["dispatch_visual", "dispatch_knowledge", "dispatch_reasoning_text_structure", "dispatch_verification", "finish"], "object_existence": ["dispatch_visual", "dispatch_reasoning_hybrid", "dispatch_verification", "finish"], "object_quantity": ["dispatch_visual", "dispatch_reasoning_text_structure", "dispatch_verification", "finish"]}
- Policy parameters changed: yes
- Cognitive Expert fingerprints unchanged: yes
- Every sampled action linked to its pre-state transition: yes
- Evaluation parameter updates: zero
- verl installed in this environment: False

The learned greedy policy skips Knowledge for quantity/existence tasks, keeps
Knowledge for airport classification, uses Text+Structure for counting, and
uses paid Hybrid latent disclosure for existence. References are available only
to the reward environment, never to Controller or Expert observations.
