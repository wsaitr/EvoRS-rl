# EvoRS-Comm 核心架构决策 v2

> 状态：Phase 0--5 的核心接口、机制与验收证据已经落地；Phase 6 论文实验待执行
>
> 文档类型：实现架构 / 设计契约；“待验证”内容仍不代表论文结论
>
> 更新日期：2026-09-03

## 0. 文档目的与优先级

本文固化 EvoRS-Comm 当前已经确定的核心设计，作为后续模块设计、实现、论文撰写和实验规划的共同依据。

本文只冻结机制边界，不提前锁死尚未验证的算法细节。后续每次讨论应更新本文对应章节，而不是在多个文档中维护互相冲突的版本。

本文使用以下状态语义：

- **已确定**：已经达成共识的设计边界；
- **计划实现**：方向明确但当前代码尚未完成；
- **待验证假设**：必须由实验支持，也允许得到否定结论；
- **开放问题**：尚未做最终选择；
- **暂缓**：不进入当前阶段。

当本文与以下旧文档在 Agent 体系、运行时、Workspace、通信或 RL 路线方面冲突时，以本文为准：

- `GUIDE.md`
- `FRAMEWORK_DECISION.md` 的旧版 Stage 1/2 框架选择
- `AI_IMPLEMENTATION_PROMPT.md`
- 根目录 `README.md` 中的旧三阶段架构说明

旧文档仍可作为数据集、硬件适配、部署经验和早期实现记录使用。

---

## 1. 论文定位

EvoRS-Comm 面向单图及大幅遥感图像理解任务，采用两个相互独立但协同工作的机制：

1. **单 Agent 内部演化（Intra-Agent Evolution）**：专家在自身局部任务内，通过候选生成、工具反馈、评价、变异或修正、选择，逐步改进输出。
2. **多 Agent 动态协作（Dynamic Collaboration）**：控制器根据当前协作状态、证据、不确定性和预算，动态决定激活哪个专家、向其披露哪些信息以及何时结束任务。

层次化空间表示与多模态通信连接这两个机制：专家产生的文本、结构和视觉表示绑定到问题相关的空间节点，控制器再将其中必要的部分路由给后续专家。

当前建议的一句话定义为：

> **EvoRS-Comm 是一个面向遥感视觉理解的双层协作框架：专家内部通过受预算约束的演化搜索改进局部决策，专家之间通过基于共享状态的动态路由和层次化多模态通信完成高效协作。**

### 1.1 明确非目标

本项目明确不做：

- 不让 Agent 自主发明不可解释的新通信语言；
- 不为单张图片构建完整、全量的地理世界模型；
- 不在每个任务中无条件启动全部 Agent；
- 不允许 Agent 直接调用另一个 Agent；
- 不让所有 Agent 自动读取完整共享状态；
- 不把 latent handle 或 bbox 字符串冒充真实视觉 latent；
- 不在真实训练闭环完成前宣称提出了新的 RL 算法；
- 不把预期的性能提升写成已经成立的实验结论。

---

## 2. 总体架构：两个嵌套循环

```text
外层：多 Agent 动态协作

Workspace W_t
    -> ControllerObservation
    -> EvoRSController / RoutingPolicy
    -> Dispatch Expert
    -> AgentResult
    -> Validate + Reduce
    -> Workspace W_{t+1}
    -> Continue | Finish

内层：单 Agent 自身演化

AgentObservation
    -> Seed Candidates
    -> ReAct / Tool Feedback
    -> Fitness Evaluation
    -> Variation / Revision
    -> Selection
    -> AgentResult
```

两个循环必须保持边界：

- 外层控制器不承担领域推理，不直接看图，不生成事实和答案；
- 内层专家不选择下一个 Agent，不直接修改共享状态，不直接调用其他 Agent；
- 专家内部未被选中的候选和推理草稿属于私有状态，不进入共享 Workspace；
- 只有最终选中的结构化结果可以提交给外层运行时。

---

## 3. 单 Agent 内部演化

### 3.1 已确定语义

“演化”专指单个专家内部的局部搜索与改进，不用于描述多 Agent 路由。

每个专家可以维护若干候选个体：

- Visual Expert：候选 ROI、检测集合、尺度或视觉证据方案；
- Reasoning Expert：候选假设、关系解释或答案方案；
- Verification Expert：候选核验路径、反例或修复建议；
- Knowledge Expert：候选领域知识解释或带来源的知识事实。

若论文使用“进化算法”这一术语，实现至少需要明确：

```text
Individual   候选方案
Variation    候选变异或修正算子
Fitness      证据、相关性、一致性与成本评价
Selection    保留较优候选
Generation   有限预算内的迭代轮次
```

如果最终实现只有一次 ReAct 或简单自反思，而没有候选、变异和选择，应改称“迭代自修正”，不声称使用进化算法。

### 3.2 初始实现候选

首版可考虑轻量的 `(1 + lambda)` 策略：保留一个当前最佳候选，每轮产生少量变体，评价后选择下一轮父候选。

该选择尚未冻结。以下内容后续单独讨论：

- 候选表示；
- 变异算子；
- 是否需要交叉；
- 各专家的 fitness；
- 每次调用的 population 与 generation；
- 内部演化是否以及如何参与 RL 联合训练。

---

## 4. Cognitive Agent Pool

专家池按认知能力划分，而不是按固定图像空间范围划分。

| 专家 | 主要职责 | 主要结构化产物 |
|---|---|---|
| Visual Expert | 图像观察、ROI 发现、检测、视觉特征提取 | `Evidence`、`SpatialNode`、`ArtifactRef` |
| Reasoning Expert | 空间、语义和多步逻辑推理 | `Claim`、`Relation`、`AnswerCandidate` |
| Verification Expert | 矛盾检查、证据核验、不确定性分析 | `Verdict`、`EvidenceRequest`、状态更新建议 |
| Knowledge Expert | 遥感领域知识与概念关系 | 带来源的 `KnowledgeFact`、解释建议 |

### 4.1 Controller 不属于 Agent Pool

外层控制组件统一命名为：

- `EvoRSController`：运行时控制器；
- `RoutingPolicy`：控制器内部的决策策略；
- `RuleRoutingPolicy`：规则式基线；
- `LearnedRoutingPolicy`：后续学习式策略。

论文叙述中可称“协作控制器”或“动态路由策略”。“演化”一词保留给专家内部机制，避免概念重叠。

---

## 5. EvoRSController 与 RoutingPolicy

### 5.1 Controller 观察

Controller 不读取整张图片、完整 latent 或所有原始历史，只读取 Workspace 的紧凑摘要：

```text
ControllerObservation
├── task_summary
├── spatial_summary
├── evidence_summary
├── claim_and_answer_summary
├── agent_status_summary
├── communication_summary
└── remaining_budget
```

### 5.2 Controller 动作

首版外层动作保持最小：

```text
DispatchAction
├── agent_id
├── objective
├── focus_node_ids
├── context_spec
└── local_budget

FinishAction
└── answer_candidate_id

AbortAction
└── reason
```

`context_spec` 同时定义本次激活所携带的通信上下文：

```text
ContextSpec
├── spatial_level
├── modalities          # text / structure / latent
├── payload_refs        # Workspace 中已提交的记录引用
└── max_items
```

工具选择属于当前专家内部行为，不进入首版 Controller 动作空间。

### 5.3 专家输出

专家不能直接修改 Workspace，只能返回：

```text
AgentResult
├── workspace_delta
├── produced_artifacts
├── answer_candidate
├── confidence
├── uncertainty
├── capability_request
└── metrics
```

`capability_request` 表达还需要什么能力或证据，例如“需要重新观察 Region_07”，不能写死“调用 Visual Expert”。下一步由 Controller 决定。

### 5.4 最终答案所有权

- Reasoning Expert 产生 `AnswerCandidate`；
- Verification Expert 可以验证、拒绝或提出补证据要求；
- Controller 只能通过 `FinishAction(answer_candidate_id)` 选择一个已存在且满足终止条件的候选；
- Controller 本身不得生成任务答案。

---

## 6. Runtime 与 LangGraph

LangGraph 只承担运行时职责，不承载论文领域逻辑。

目标拓扑：

```text
START
  -> InitializeState
  -> BuildControllerObservation
  -> EvoRSController
       -> FINISH -> Finalize -> END
       -> DISPATCH
            -> ContextRouter
            -> Selected Expert Subgraph
            -> ValidateDelta
            -> WorkspaceReducer
            -> MeasureProgressAndCost
            -> EvoRSController
```

每个专家可实现为独立子图：

```text
Observe
  -> SeedCandidates
  -> Reason / Act
  -> ToolRuntime
  -> Evaluate
  -> EvolveAgain | EmitResult
```

首版每个 environment step 只激活一个专家。并行激活留作后续扩展，以保持状态转移、通信因果和 RL 轨迹清晰。

运行时需要支持：

- 条件路由；
- 状态 reducer；
- step 与 budget 终止；
- 超时、失败、非法引用与重试；
- 确定性 replay；
- 轨迹与成本记录。

参考：<https://docs.langchain.com/oss/python/langgraph/graph-api>

---

## 7. Shared Workspace 与通信

### 7.1 已确定决策

采用：

> **受控共享 Workspace + 显式消息路由。**

不采用全共享黑板，也不采用纯消息复制。

核心区分：

```text
Workspace      系统已经保存了什么
Communication 哪个 Agent 在何时、以什么模态看到其中哪一部分
Controller    决定何时向谁披露哪一部分
```

Workspace 是单个 `image + question` episode 的临时协作状态，不是长期记忆或大型知识库。

### 7.2 状态边界

```text
EvoRSState
├── Shared Workspace
│   ├── Task Context
│   ├── Hierarchical Spatial Representation
│   ├── Evidence
│   ├── Claims / Answer Candidates
│   └── Artifact Handles
│
├── Communication State
│   ├── Message Log
│   ├── Agent Inboxes
│   └── Access Grants
│
└── Control State
    ├── Agent Status
    ├── Current Assignment
    ├── Budget
    ├── Step
    └── Termination Status
```

此外还有两个不直接放入 Workspace 的层：

- `Private Scratchpad`：单个 Agent 内部候选与临时推理状态；
- `ArtifactStore`：图片、crop、global latent、ROI latent 等重数据。

### 7.3 可见性规则

- Task Context 对所有被激活专家可见；
- Agent 自己尚未提交的候选仅在私有 scratchpad 可见；
- 其他 Agent 尚未验证的结果，只有经过消息路由后才对接收者可见；
- 已验证事实可以进入公共索引，但 ObservationBuilder 仍按当前任务和空间范围筛选；
- crop 与 latent 只保存 handle，只有消息指定相应模态且接收端兼容时才加载；
- Controller 只看摘要、状态、不确定性和成本，不读取完整 latent；
- Agent 读取共享状态的唯一入口是 `ObservationBuilder`，不能绕过可见性规则直接读取完整 Workspace。

形式上：

```text
Environment State: W_t
Agent Observation: o_t^i = phi_i(W_t, inbox_i, access_i)
```

消息动作必须实际改变接收 Agent 的 observation。删除一条消息后，如果接收方仍能看到完全相同的信息，则该消息不构成有因果作用的通信。

### 7.4 写入规则

- Agent 只返回 `WorkspaceDelta`，不能原地修改共享状态；
- Evidence、Message 和 Artifact 元数据采用追加式记录；
- Reducer 负责 schema 校验、引用校验、版本检查和冲突处理；
- Router 只能投递已经提交成功的记录引用；
- 所有共享记录必须带 `created_by`、step 和 provenance；
- `FINISH` 必须引用已经存在的 `AnswerCandidate`。

### 7.5 通信语义

物理上信息只保存一次；逻辑消息记录“谁将哪些信息以何种形式披露给谁”。

```text
MessageEnvelope
├── message_id
├── producer_agent
├── routed_by
├── receiver_agent
├── purpose
├── spatial_scope
├── modalities
├── record_refs
├── optional_summary
├── communication_cost
└── step
```

消息应优先传引用，不重复复制节点、bbox 或 tensor。Text 可以作为简短摘要存在。

即使物理实现只传 handle，通信成本仍按接收方实际加载的信息计量：

- Text：token 数；
- Structure：序列化字节或节点/关系数；
- Latent：维度、dtype 与实际字节；
- 另行记录 Agent 调用、Tool 调用和延迟等计算成本。

### 7.6 论文中的正确定位

Workspace 本身不是独立创新点。论文方法应表述为：

> **基于遥感层次空间索引的、策略控制的信息披露与多模态通信机制。**

其中：

- 层次空间表示定义信息所属的空间范围；
- Workspace 提供单一事实源；
- Message 定义接收者、范围和模态；
- Controller 决定何时披露哪些信息。

---

## 8. 轻量层次空间表示

### 8.1 设计决策

不实现完整 `Spatial World Model`。采用：

> **面向问题、按需增量构建的轻量层次空间表示。**

它只描述当前问题需要的暂态空间证据，不声称完整重建真实世界。

```text
Scene
├── Region     # 可选
├── Group      # 可选
└── Object     # 可以跳级挂接

Sparse Relations, 按问题生成
├── inside
├── near
├── left_of
├── aligned_with
└── connected_to
```

约束：

- Region、Group 和 Object 允许跳级；
- 禁止为了满足固定四层格式制造空节点；
- `parent_id` 只表达主要包含关系；
- 非包含关系放在可选的稀疏 relation 表；
- 默认只创建与问题相关的节点和 ROI，不做全图全量场景图解析；
- Group 仅在压缩重复对象或表达功能组合有价值时创建。

### 8.2 最小节点

```text
SpatialNode
├── node_id
├── level
├── semantic
├── bbox
├── confidence
├── parent_id
├── evidence_refs
├── visual_ref
├── provenance
└── status
```

rotated bbox、polygon、GSD、地理坐标、传感器和时间等字段只有在真实数据集提供且任务需要时再加入，不自行伪造。

---

## 9. Image / ROI Latent Communication

### 9.1 与空间节点绑定

- Scene Node 可以引用 global image latent；
- Region/Object Node 可以引用 ROI latent；
- Group Node 可以选择性引用成员 latent 的 pooled representation；
- Workspace 和 Message 只保存 `visual_ref`；
- tensor 实体保存在 ArtifactStore。

### 9.2 Artifact 描述

```text
VisualArtifact
├── artifact_id
├── encoder_id
├── encoder_layer
├── projection_id
├── source_roi
├── shape
├── dtype
├── normalization
├── checksum
├── byte_size
└── compatible_consumers
```

### 9.3 MVP 边界

- 首版所有专家共享同一 vision encoder / projection；
- latent 按需生成、缓存并通过 handle 引用；
- 接收方不兼容时回退到 crop、structure 或 text；
- latent channel 必须 feature flag 控制；
- 必须证明接收专家真实消费了 latent；
- latent 不阻塞 Text + Structure MVP。

以下内容后续单独讨论：

- 选取哪一层视觉表示；
- projector 结构；
- 接收方采用 visual token、cross-attention 还是其他融合方式；
- 同构与异构 Agent；
- latent 压缩和训练目标。

---

## 10. RL 定位

### 10.1 已确定边界

多 Agent 层面的学习目标是动态路由与策略学习，不称为“多 Agent 间演化”。

首个可训练版本建议：

- 冻结 Visual、Reasoning、Verification、Knowledge 专家；
- 只训练 `LearnedRoutingPolicy`；
- 优化 Agent 选择、信息披露、局部预算和终止；
- 工具选择仍由专家内部完成；
- 稳定后再讨论联合训练专家内部演化。

在只有一个集中式 RoutingPolicy 被训练时，论文应称为：

> **基于强化学习的集中式动态协作优化。**

只有在定义多个专家策略、局部观察、联合状态和信用分配并完成真实联合训练后，才升级为严格意义上的 Multi-Agent RL。

### 10.2 环境与动作

```text
Environment State = 完整 EvoRSState
Controller Observation = Workspace 紧凑摘要
Controller Action = Dispatch | Finish | Abort
Expert Local Action = ToolRequest | WorkspaceDelta | CapabilityRequest
```

无效动作需要 mask，例如：

- 没有候选答案时禁止 Finish；
- 没有 latent 时禁止选择 latent 模态；
- 禁止引用不存在或未授权的节点；
- 禁止超出 Agent、Tool、token、latency 或通信预算；
- 禁止无进展的重复动作。

### 10.3 Reward 方向

证据质量和结构一致性如果定义为正向质量，应使用正号：

```text
r_t = terminal_task_score
    + evidence_quality_gain
    + structure_quality_gain
    - message_cost
    - agent_cost
    - tool_cost
    - redundancy_cost
    - invalid_action_penalty
```

建议使用增量质量而不是每步重复奖励总量，避免刷分。

### 10.4 当前训练实现与 verl 边界

当前 7 动作、21 特征的独立离散 Controller 使用直接 on-policy policy gradient
训练；这比将离散路由动作伪装成语言 token 更直接，也能完整记录动作概率、
mask、奖励和因果 transition。`verl` 适配层负责输出后续 token-policy 训练需要的
数据与 reward 契约，但当前不宣称已经执行 verl 训练。

适配关系：

```text
verl
  -> EvoRSAgentLoop
  -> LearnedRoutingPolicy 生成结构化控制 action
  -> LangGraph / EvoRSState 执行动作
  -> RewardManager 根据答案、轨迹、状态增量和成本评分
```

若 Phase 6 将 Controller 扩展为 LLM token policy，再以同一 Workspace、动作和
RewardManager 契约接入 verl；PPO、GRPO 或多 Agent 扩展算法须在实际实现和实验后命名。

参考：<https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst>

---

## 11. 示例：统计飞机数量

```text
1. Controller
   -> Dispatch Visual Expert
   -> scope=Scene, local evolution budget=2 generations

2. Visual Expert 内部演化
   -> 产生多个候选 ROI / 检测集合
   -> 依据证据和成本选择较优候选

3. Visual Expert 返回 AgentResult
   -> Region / Group / Object 节点
   -> Detection Evidence
   -> ROI latent_ref
   -> 不向 Workspace 提交 Visual 私有答案

4. Reducer 提交；Controller 重新决策
   -> Dispatch Reasoning Expert
   -> 由策略选择 Text+Structure 或付费 Hybrid 披露

5. Reasoning Expert
   -> 基于被授权的节点、证据与可选 latent 推理
   -> 产生 Claim 与未验证 AnswerCandidate(count=6)

6. Controller Dispatch Verification Expert
   -> 只披露候选答案及其证据依赖
   -> 检查漏检、重复框和计数一致性
   -> 输出 Verdict 与 Verified AnswerCandidate

7. Controller
   -> Finish(verified_answer_candidate_id)
```

Knowledge Expert 不因存在于池中而自动运行。

---

## 12. 工程目录方向

目标结构保留现有 `src` 布局：

```text
src/rs_hmcomm/
├── agents/
│   ├── base.py
│   ├── visual.py
│   ├── reasoning.py
│   ├── verification.py
│   ├── knowledge.py
│   └── intra_agent_evolution.py
├── runtime/
│   ├── graph.py
│   ├── controller.py
│   └── policies/
│       ├── base.py
│       ├── rule.py
│       └── learned.py
├── workspace/
│   ├── state.py
│   ├── reducer.py
│   └── observation.py
├── spatial/
│   ├── hierarchy.py
│   └── relations.py
├── communication/
│   ├── schema.py
│   ├── router.py
│   └── cost.py
├── artifacts/
│   └── store.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── crop.py
│   ├── detection.py
│   └── encoder.py
├── environment/
│   ├── env.py
│   ├── actions.py
│   └── rewards.py
├── rl/
│   └── verl_adapter.py
├── backends/
├── datasets/
├── data/
└── eval/
```

旧版 Global/Local/Hierarchy/Residual Agent 建议保留为固定空间工作流 baseline，而不是直接删除。

---

## 13. 分阶段路线

### Phase 0：契约与可观测性

- 定义 State、Action、AgentResult、WorkspaceDelta、MessageEnvelope；
- 定义预算、成本、事件日志与 replay；
- 修正实验 baseline 的隔离性。

### Phase 1：单 Generalist Agent

- 先实现一个能够端到端完成任务的 Generalist Agent；
- 跑通 image + question -> ReAct / tools -> evidence -> answer；
- 建立 Single VLM 与 Single ReAct Agent baseline；
- 保持 `AgentObservation` / `AgentResult` 接口可被后续专家复用；
- 此阶段不引入多 Agent Controller。

### Phase 2：单 Agent 内部演化

- 在 Generalist Agent 外实现可插拔的 Intra-Agent Evolution；
- 明确 candidate、variation、fitness、selection 和 generation；
- 与普通 ReAct、重试、Best-of-N 和迭代自修正做等预算比较；
- 失败候选保留在 Private Scratchpad，不进入共享状态。

### Phase 3：EvoRS 特有表示层

- 受控 Workspace 与 WorkspaceDelta / Reducer；
- 问题驱动的轻量层次空间表示；
- ArtifactStore；
- global / ROI latent 的生成、引用和消费；
- Text / Structure / Latent 的 scoped observation；
- 先由单 Agent 完成写入、重新读取和使用闭环。

### Phase 4：无 RL 的多 Agent 协作

- Cognitive Agent Pool；
- LangGraph 顶层运行时；
- EvoRSController + RuleRoutingPolicy；
- ContextRouter、MessageEnvelope 与 Agent Inbox；
- 受控信息披露和通信成本；
- 固定流程、全 Agent 和规则路由 baseline；
- 此阶段不引入学习式路由。

当前开发门禁结果（20 个 paired frozen-perception 样本）：Direct VLM 50%，
完整 Single Agent 60%，Multi Agent 70%；完整证据和统计限制见
`EVORS_PHASE4_COMPARISON_SPEC.md` 与
`evidence/2026-09-02_phase4_comparison/`。该结果尚不构成论文级显著性结论。

### Phase 5：学习式动态路由

- LearnedRoutingPolicy；
- EvoRSAgentLoop / verl adapter；
- 冻结专家，先训练 Controller；
- 稳定后再评估联合训练。

当前实现状态（2026-09-03）：

- 已采用带有效动作 mask 的 7 动作、21 特征离散 `LearnedRoutingPolicy`；
- 已实现 on-policy policy gradient、状态条件基线、熵正则、梯度裁剪和 JSON checkpoint；
- 每个采样 action 在 LangGraph transition 发生前产生，并通过 `policy_action_id` 一一审计；
- 720 个确定性机制训练 episode 后，冻结 Expert 指纹不变，策略参数 SHA-256 发生变化；
- 6 个 held-out fixture 上，计数/存在性路径为 Visual→Reasoning→Verification，
  分类路径为 Visual→Knowledge→Reasoning→Verification，准确率为 100%；
- 计数和分类选择 Text+Structure，只有依赖视觉表示的存在性任务选择付费 Hybrid；
  平均奖励由 0.2357 提升到 0.5690；
- 该结果只证明 RL 动态路由机制闭环，不是论文 benchmark 结论；
- verl 已实现 record/reward/AgentLoop 顺序适配，但当前独立离散策略直接训练更合理，
  不宣称已执行 verl token-policy 训练。

完整契约和证据见 `EVORS_PHASE5_RL_SPEC.md` 与
`evidence/2026-09-03_phase5_routing/`。

### Phase 6：完整 Benchmark

- VRSBench；
- CHOICE；
- XLRS-Bench；
- GEO-Bench-VLM；
- 多 seed、置信区间和 accuracy-cost Pareto。

---

## 14. 最低验收标准

实现需要满足以下可验证行为：

- [x] Controller 不产生视觉事实、知识事实或任务答案；
- [x] 专家不能直接调用另一个专家；
- [x] 专家不能原地修改共享 Workspace；
- [x] RuleRoutingPolicy 与 LearnedRoutingPolicy 使用同一接口；
- [x] 每个专家只收到 ObservationBuilder 构造的 scoped view；
- [x] 删除一条消息会改变对应接收者可见的信息；
- [x] Workspace 中不直接保存图片、crop 或 latent tensor；
- [x] Region、Group、Object 可以按任务跳级；
- [x] 不为满足四层格式创建无证据节点；
- [x] latent 实验能证明接收方真实消费了 latent；
- [x] 通信成本统计实际披露的 text、structure 和 latent；
- [x] Finish 必须引用已存在且 VERIFIED 的 AnswerCandidate；
- [x] Episode 可以从 transition/message 日志确定性回放；
- [x] 每个 RL action 在采样时真实影响后续状态，而不是 episode 结束后补记；
- [x] 固定工作流、规则路由和学习路由可以在相同预算下比较。

---

## 15. 待验证研究假设

以下内容是论文需要检验的假设，不是预先成立的结论：

- **H1**：在相同通信与计算预算下，层次结构通信优于纯文本或 flat ROI 通信；
- **H2**：ROI latent 在 Text + Structure 基础上带来额外任务收益；
- **H3**：动态路由相较固定流程或全 Agent 激活具有更好的准确率—成本权衡；
- **H4**：单 Agent 内部演化在相同推理预算下优于普通 ReAct、重试或迭代自修正；
- **H5**：LearnedRoutingPolicy 在未见测试集上优于 RuleRoutingPolicy；
- **H6**：受控信息披露优于完整 Workspace 全量可见和纯消息复制。

如果实验不支持某项假设，应报告无显著收益、失败条件或适用范围，不能通过事后更换成本口径回避否定结果。

---

## 16. 必做消融

### 15.1 Workspace / Communication

1. Full Blackboard：所有专家自动看到完整 Workspace；
2. Message Only：没有共享事实源，消息复制 payload；
3. Gated Hybrid：受控 Workspace + 引用式消息。

### 15.2 空间表示

1. Text only；
2. Flat ROI list；
3. 刚性四层树；
4. 可跳级轻量层次表示；
5. 轻量层次表示 + sparse relations。

### 15.3 通信模态

1. Text；
2. Text + Structure；
3. Text + Structure + Crop Reference；
4. Text + Structure + ROI Latent；
5. Global Latent + ROI Latent。

### 15.4 协作与内部演化

1. 固定全 Agent；
2. 固定 workflow；
3. 随机路由；
4. 规则路由；
5. 学习式路由；
6. 无内部演化；
7. 内部迭代自修正；
8. 真实候选-变异-选择演化。

统一报告：任务分数、Agent/Tool 调用、文本 token、结构字节、latent 字节、节点与关系数、重复信息、延迟、峰值内存和 accuracy-cost Pareto。

---

## 17. 与当前代码的迁移关系

当前仓库是旧版原型，不能视为本文目标架构已经实现。

可复用：

- `SceneNode` / `SceneTree`：轻量层次表示的起点；
- `AgentMessage` / `MessageBus`：消息 schema 与统计的起点；
- `RuleController`：`RuleRoutingPolicy` baseline 的起点；
- backends、datasets、data、eval、device 与部署基础；
- 现有测试中的树、消息、日志和奖励用例。

需要替换或重构：

- Global/Local/Hierarchy/Residual 主调用链；
- 自写固定 orchestrator 循环；
- Agent 直接修改 `SceneTree`；
- 所有 Agent 自动收到完整 tree；
- Agent 硬编码下一个 receiver；
- episode 结束后补采样的通信动作；
- 仅生成配置但不执行训练的 verl scaffold。

旧 Agent 体系应迁移到 baseline，而不是作为新主架构继续扩展。

---

## 18. 已冻结、暂缓与开放问题

### 18.1 已冻结的机制边界

- 演化发生在单 Agent 内部；
- 多 Agent 层面采用动态路由与策略学习；
- Controller 位于控制层，不属于 Cognitive Agent Pool；
- Cognitive Agent Pool 包含 Visual、Reasoning、Verification、Knowledge；
- Agent 之间不直接调用；
- 采用受控 Workspace + 显式消息路由；
- 空间表示是问题驱动、按需构建的轻量可跳级层次；
- latent 绑定空间节点并通过 ArtifactStore 引用；
- latent 不阻塞 Text + Structure MVP；
- 首个 RL 版本优先训练 RoutingPolicy，专家冻结；
- LangGraph 作为 runtime，verl 作为目标训练框架。

### 18.2 暂缓实现

- 异构模型 latent 对齐；
- 多专家并行激活；
- 专家与 Controller 联合训练；
- 完整地理关系图或长期世界模型；
- 跨 episode 长期记忆；
- 自动新增或删除 Agent 类型；
- 新 RL 算法命名与理论声明。

### 18.3 后续逐项讨论

1. 单 Agent 演化的 candidate、variation、fitness 和 selection；
2. RuleRoutingPolicy 的第一版规则；
3. ControllerObservation 的具体序列化；
4. Workspace schema、Reducer 与访问控制；
5. 轻量层次节点和 sparse relation 的最终字段；
6. MessageEnvelope、通信成本和 access grant；
7. Visual latent 的层、projector 和接收端融合；
8. Knowledge Expert 的知识来源与检索方式；
9. 若 Phase 6 扩展 token policy，是否从当前独立小网络迁移到 verl AgentLoop；
10. 联合训练专家时采用 PPO、GRPO 或其他算法；
11. Benchmark 子集、预算定义和主评价指标；
12. 论文贡献数量与最终措辞。

---

## 19. 决策索引

后续逐项讨论时使用以下 ID，避免只依赖章节或口头描述：

| ID | 决策 | 状态 |
|---|---|---|
| D-001 | 单 Agent 内部演化与多 Agent 动态路由严格分离 | 已确定 |
| D-002 | `EvoRSController` 位于控制层，不属于 Cognitive Agent Pool | 已确定 |
| D-003 | Controller 内部采用可替换的 `RoutingPolicy` | 已确定 |
| D-004 | Agent Pool 使用 Visual、Reasoning、Verification、Knowledge 四类专家 | 已确定 |
| D-005 | Agent 之间不直接调用 | 已确定 |
| D-006 | 采用受控 Workspace + 显式消息路由 | 已确定 |
| D-007 | Private Scratchpad 不进入共享状态 | 已确定 |
| D-008 | 采用问题驱动、可跳级的轻量层次空间表示 | 已确定 |
| D-009 | 不建设完整 Spatial World Model | 已确定 |
| D-010 | crop 与 latent 实体存放在 ArtifactStore | 已确定 |
| D-011 | Message 主要传引用，但按实际披露载荷计费 | 已确定 |
| D-012 | LangGraph 作为 runtime | 已确定 |
| D-013 | 规则与学习路由共用策略接口 | 已确定 |
| D-014 | 首个 RL 版本冻结专家、优先训练 RoutingPolicy | 当前推荐 / 待实验确认 |
| D-015 | Phase 5 独立小型离散 Controller 使用直接 policy gradient；verl 保留 token-policy adapter | 已实现 |
| D-016 | 专家联合训练的 PPO/GRPO 等具体算法 | 开放问题 |
| D-017 | 异构 latent 对齐、完整关系图和并行专家 | 暂缓 |

---

## 20. 当前核心结论

```text
EvoRS-Comm
= Intra-Agent Evolution
+ Workspace-Conditioned Dynamic Routing
+ Controlled Shared Workspace
+ Hierarchy-Guided Multimodal Information Disclosure
+ Text / Structure / ROI Latent Communication
```

这份文档冻结的是上述机制之间的职责与边界。具体数据结构、算法超参数、模型、训练框架适配和实验设置将在后续讨论中逐项确定。
