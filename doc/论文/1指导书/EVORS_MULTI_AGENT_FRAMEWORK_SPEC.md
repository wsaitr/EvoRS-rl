# EvoRS 无 RL 多 Agent 框架完善规格

> 状态：实现完成，真实模型双路径 smoke 已通过
>
> 日期：2026-09-01

## Context

Phase 4 已有 LangGraph、四类 Cognitive Expert、Workspace、引用式消息和三种
规则 baseline，但当前稳定入口主要验证脚本化控制流。控制器仍能持有完整
Workspace，通信上下文由 Runtime 根据记录 ID 前缀推断，真实 VLM 专家也没有
形成一条独立、可复验的 Phase 4 门禁。

本规格只完善无 RL 的多 Agent 协作。`Evolution Controller` 固定为控制层组件；
“演化”只描述单个 Expert 激活内部的 candidate、variation、fitness 和 selection。

## Goals

- Controller 只观察紧凑状态摘要，不读取图像、原始证据文本或 latent tensor；
- RoutingPolicy 显式产生 `DispatchDecision + ContextSpec`，由 Router 执行披露；
- Workspace 是 episode 的单一事实源，消息只传已提交记录的引用和模态；
- 每个 step 只激活一个 Expert，Expert 只返回 `WorkspaceDelta`；
- 规则路由根据任务和当前状态动态跳过无必要 Expert；
- 真实 Qwen VLM 的 Visual→Reasoning→Verification 路径可运行、回放和计费；
- 保留 Fixed、All-Agent、Rule 三种非 RL baseline。

## Non-Goals

- 不实现 PPO、GRPO、MAGRPO、verl 或参数更新；
- 不把 Controller 放进 Cognitive Agent Pool；
- 不允许 Agent 直接调用另一个 Agent；
- 不把 Workspace 表述为独立论文创新点；
- 不让 Workspace 保存 image/crop bytes 或 latent tensor values；
- 不把单样本 smoke 表述为准确率结论。

## Architecture Contract

```text
Workspace W_t
  -> ControllerSummaryBuilder
  -> EvolutionController / RoutingPolicy
  -> Dispatch(expert, objective, context_spec, local_budget)
  -> ContextRouter (refs + modalities)
  -> ObservationBuilder
  -> exactly one Cognitive Expert
  -> validate WorkspaceDelta
  -> WorkspaceReducer -> W_t+1
  -> Controller
```

控制层、共享层和私有层边界：

- Controller：只做 dispatch / finish / abort；
- Workspace：保存已提交的 Evidence、Claim、AnswerCandidate、SpatialNode 和
  Artifact descriptor；
- Message/Inbox：记录某个 Expert 在某一步被披露了哪些 Workspace refs；
- ArtifactStore：保存图像、crop 和 latent 的物理内容；
- Expert private trace：保存未选中的候选和内部演化过程。

## Requirements

1. `ControllerObservation.workspace` 必须是不可反查原始 payload 的紧凑摘要；
2. 摘要至少包含版本、各类记录数、可路由记录描述、候选状态和通信/成本概况；
3. `DispatchDecision` 至少包含 expert、objective、`ContextSpec` 和可选 local budget；
4. `ContextSpec` 明确 modalities、payload refs、focus nodes、spatial level 和 max items；
5. Runtime 不得根据 ID 命名约定推断上下文；只执行 policy 返回的 spec；
6. Router 必须拒绝未知 ref、超过 max items 的披露和未授权 latent；
7. Task Context 对被激活 Expert 始终可见，其余记录只经 Inbox/Grant 可见；
8. Rule policy 根据 Workspace 状态而非仅按历史序列推进；
9. counting 默认不激活 Knowledge；classification/domain task 可激活 Knowledge；
10. Reasoning 前必须存在可见 Evidence，Verification 前必须存在 proposed candidate；
11. Finish 只能引用现有 `VERIFIED` AnswerCandidate；
12. Expert 异常、非法 delta、预算不足和无进展必须成为可回放终止状态；
13. 真实 VLM expert 不读取 GT、reference answer 或 `mock_answer`；
14. 真实运行至少记录 agent path、messages、workspace versions、token/通信成本和答案。

## Acceptance Criteria

- [x] Controller observation 不再包含 `WorkspaceState`、图片对象或 latent values；
- [x] Dispatch 序列化中包含完整 `ContextSpec`；
- [x] 删除 routed message 后，目标 Expert 的可见记录发生变化；
- [x] counting 路径为 Visual→Reasoning→Verification，Knowledge 未被激活；
- [x] airport classification 路径包含 Knowledge；
- [x] Runtime 每次只执行一个 Expert，且 Workspace 每次成功执行只增加一个版本；
- [x] Controller 不能 finish proposed/missing candidate；
- [x] Fake VLM 完整多 Agent 集成测试通过；
- [x] 本地真实 Qwen 至少完成一个无标签泄漏的 Phase 4 episode；
- [x] Phase 4 新测试和项目全量测试通过。

## Verification Result

2026-09-01 使用固定本地 `evors/qwen3.8-27b-q4` 在
`vrsbench-val-001` 上完成两个真实 episode。Task Context 只包含 image、question、
task type、answer format 和样本 ID；reference answer 只在 episode 结束后评分。

| Path | Answer | Agent path | Inner evolution | Objects | Latent communication |
|---|---:|---|---|---:|---:|
| Text + Structure | 6 | Visual→Reasoning→Verification | 4 candidates / 3 variations / 3 generations | 6 | 0 bytes |
| Qwen native latent | 6 | Visual→Reasoning→Verification | 4 candidates / 3 variations / 3 generations | 6 | 143,360 bytes |

两条路径均为 Workspace version 3、2 条显式 routed message、6 次真实模型调用。
latent 路径提交 7 个 Qwen-native descriptor；Visual 私有循环实际加载
143,360 bytes，Reasoning 从授权 scoped view 消费后产生 13 项诊断。

Verifier 的单图结果曾错误尝试把证据一致的 6 改为 0。最终验证策略不依赖
自报置信度：当多视图答案、Reasoning candidate 与 Workspace 的 Object 数一致时，
单次 whole-image contradiction 只能被记录，不能覆盖三方一致的结构化证据。

全量离线/本地测试结果：`328 passed, 3 skipped`。

证据：`evidence/2026-09-01_multi_agent_framework/`。

该结果证明 Phase 4 的机制闭环和单样本真实执行，不构成多 Agent 已统计优于
单 Agent/VLM 的论文结论；准确率、成本 Pareto、多 seed 和消融属于 Phase 6。

## Implementation Tasks

1. 新增紧凑 `ControllerWorkspaceSummary` 和统一 summary builder；
2. 新增 `ContextSpec`，扩展 dispatch/observation/transition 契约；
3. 将 Rule/Fixed/All/Learned policy 改为显式选择 refs 与 modalities；
4. 将 Runtime 中的 ID 前缀披露替换为 `ContextSpec` 执行；
5. 强化 finish、latent consumer、max-items 和预算验证；
6. 增加真实 VLM Phase 4 runner，并保留 scripted baseline runner；
7. 补充契约、路由、runtime、fake-backend 集成和真实 Qwen smoke 证据。

## Verification

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_control_policies.py tests/test_routed_communication.py
python -m pytest -q tests/test_langgraph_multi_agent_runtime.py tests/test_vlm_mechanisms.py
python scripts/run_real_multi_agent.py --help
python -m pytest -q
```
