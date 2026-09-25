# EvoRS 完整单 Agent 能力规格

> 状态：实现完成，真实模型 smoke 已通过
>
> 日期：2026-09-01

## Context

Phase 1–3 的机制目前分别存在：Generalist ReAct、主动感知、单 Agent
内部演化、Workspace、轻量空间层次、ArtifactStore 和 latent 编解码。
缺口是这些机制尚未在同一个真实 episode 中形成因果闭环。

本规格把论文的“专家内部演化”和“遥感特有表示层”落实到单个
Generalist Agent；多 Agent 协作仍由后续 Controller/Router 负责。

## Goals

- 同一个 Qwen VLM 完成图像观察、工具增强、结构化证据和最终答案；
- 在一次 Agent 激活内真实执行 candidate、variation、fitness、selection；
- 将选中结果通过 `WorkspaceDelta + WorkspaceReducer` 写入 episode Workspace；
- 按问题构建 Scene/Group/Object 可跳级的轻量空间描述；
- 将 image/crop/latent 实体存入 ArtifactStore，Workspace 只保存 descriptor；
- 从 scoped Workspace 重新读取授权记录并真实加载 latent bytes；
- 保留已通过的 Single Agent > Direct VLM 感知路径。

## Non-Goals

- 不实现 Cognitive Agent Pool、EvoRSController、Router 或 Agent 间消息；
- 不实现 RL/verl 或 learned routing；
- 不把 PixelStats smoke 表述为论文视觉 latent；
- 不把 20 样本开发门禁表述为论文级统计结论；
- 不向运行时暴露参考答案、GT bbox 或数据集标注。

## Requirements

1. 输入只包含 image、question、task type、answer format 和预算配置；
2. 主动感知必须提供真实的新像素视图，而非重复文本推理；
3. 计数候选至少来自全图、增强图、定位或候选复核中的两个独立观察；
4. 内部演化采用受预算约束的 `(1+1)` 搜索，fitness 至少包含证据一致性、
   结构一致性、latent 支持和成本项；
5. 未选中的候选只存在于 PrivateEvolutionTrace，不进入 Workspace；
6. Workspace 至少分两次提交：感知/表示记录和最终选中答案；
7. 空间表示必须始终有一个 Scene；只有存在定位证据时才创建 Object；
   多个同类 Object 可使用 Group 压缩，不制造空 Region/Group；
8. Scene 可引用全图 artifact/latent，Object 可引用 ROI crop/latent；
9. latent 必须使用 feature flag，启用时必须提供真实 encoder；
10. latent consumer 只能消费 scoped view 中可见且兼容的 artifact，并记录
    实际读取字节数；
11. Workspace JSON 中不得出现 image bytes、crop bytes 或 tensor values；
12. 全部模型调用、工具调用、token、耗时、空间节点和 artifact descriptor
    必须可序列化和复验。

## Acceptance Criteria

- [x] 一个集成 API 返回 answer、cost、perception、evolution、Workspace 和 scoped view；
- [x] 计数 episode 真实产生至少两个 candidate、一条 variation、fitness 和 selection；
- [x] Workspace 只包含一个最终 AnswerCandidate，淘汰候选不泄漏；
- [x] Scene/Group/Object 层次由真实定位框构建并通过完整性校验；
- [x] ArtifactStore 与 Workspace descriptor/physical payload 分离；
- [x] latent 开启时从 scoped view 实际加载 bytes，并参与 fitness；
- [x] latent 关闭时 Text + Structure 路径保持可运行；
- [x] 真实 Qwen 样本跑通完整 episode，且答案不低于主动感知输出；
- [x] 新增测试和完整项目测试通过。

## Verification Result

2026-09-01 在 `vrsbench-val-001` 上使用固定本地
`evors/qwen3.8-27b-q4` 完成两次真实 episode：

| Path | Answer | Evolution | Hierarchy | Artifacts | Latent consumed |
|---|---:|---:|---|---:|---:|
| Text + Structure | 6 | 3 generations / 4 candidates / 3 variations | Scene→Group→6 Objects | 7 | 0 |
| Qwen native latent | 6 | 3 generations / 4 candidates / 3 variations | Scene→Group→6 Objects | 14 | 143,360 bytes |

两条路径的 Workspace 均为 version 2，只包含一个公共 AnswerCandidate；
learned latent 路径实际读取 35,840 个 float32，并使 fitness 的 latent
一致性分量在候选间产生差异。全量离线测试结果为 `323 passed, 3 skipped`。

证据：`evidence/2026-09-01_integrated_single_agent/`。

该验证只证明完整机制链路和单样本真实执行，不代表 latent 已带来统计性能
提升；ARM 服务器的 learned encoder smoke 和论文级消融仍按 Phase 3/6 进行。

## Implementation Plan

1. 扩展主动感知结果，公开最终候选框但不暴露图像二进制；
2. 新增 `IntegratedSingleAgentRuntime` 及配置/结果契约；
3. 将感知调用转换为 Evidence、轻量空间节点和 image/crop artifacts；
4. 通过第一条 WorkspaceDelta 提交感知与表示；
5. 从 scoped view 消费 latent，构造候选并运行 `(1+1)` 演化；
6. 只将 winner 作为 Claim/AnswerCandidate 通过第二条 delta 提交；
7. 新增 CLI、单元测试、集成测试和真实 Qwen smoke evidence。

## Verification

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_integrated_single_agent.py
python scripts/run_complete_single_agent.py --help
python -m pytest -q
```

真实模型验证使用固定本地 `evors/qwen3.8-27b-q4` 服务；learned latent
使用同一模型家族的本地 Qwen vision tower，并单独记录 encoder/layer/projector。
