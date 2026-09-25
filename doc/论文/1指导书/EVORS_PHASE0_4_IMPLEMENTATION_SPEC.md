# EvoRS-Comm Phase 0–4 实施规格

> 状态：实施中
>
> 适用分支：`codex/phases0-4`
>
> 更新日期：2026-08-20

## 1. Context

本规格把目标架构 v2 的前半部分落实为可顺序验收的工程任务：

```text
Phase 0  数据、指标、预算、日志和接口契约
Phase 1  单 Generalist Agent 端到端闭环
Phase 2  单 Agent 内部演化
Phase 3  Workspace、轻量层次、ArtifactStore 与 latent
Phase 4  无 RL 的 Cognitive Multi-Agent 协作
```

Phase 5 的 LearnedRoutingPolicy 和 Phase 6 的论文级实验不进入本轮实现，但 Phase 0–4 的接口必须为它们保留稳定边界。

开发和 CPU 验证优先使用前置 ARM 服务器及 Docker 热挂载，不提前占用 Ascend 训练服务器。代码和镜像最终需要能够从共享并行文件系统 `lws2` 直接启动。

## 2. Goals

- 在不依赖 RL 的情况下完成从单 Agent 到多 Agent 的完整因果闭环；
- 每个阶段都能独立运行、测试、记录成本并与上一阶段比较；
- 真实业务状态、私有候选、通信和重型视觉 artifact 具有清晰边界；
- Mock/CPU smoke 与真实模型后端使用相同领域接口；
- Docker 镜像只提供环境，源码通过 bind mount 或 PFS 热挂载；
- 为后续 LearnedRoutingPolicy 记录真实的 state/action/result/cost 转移。

## 3. Non-Goals

- 本轮不实现 PPO、GRPO、MAGRPO 或 verl 训练；
- 本轮不占用 Ascend 训练节点完成大模型训练；
- 本轮不实现异构 VLM latent 对齐；
- 本轮不构建完整 GIS / Spatial World Model；
- 本轮不联合训练全部专家；
- 本轮不删除旧 Global/Local/Hierarchy workflow，旧链路保留为 baseline；
- 不把 MockBackend 或随机向量结果描述为真实模型/真实 latent 验证。

## 4. Cross-Phase Contracts

### 4.1 Agent boundary

```text
AgentObservation
├── task
├── assignment
├── scoped_records
├── inbox
└── local_budget

AgentResult
├── workspace_delta
├── produced_artifacts
├── answer_candidate
├── confidence / uncertainty
├── capability_request
└── metrics
```

Agent 不得直接调用另一个 Agent，也不得原地修改共享状态。

### 4.2 State boundary

```text
Private Scratchpad  单 Agent 候选和临时 ReAct 状态
Shared Workspace    已提交的 Task/Evidence/Claim/Answer/Spatial records
Communication       定向披露记录与 inbox
ArtifactStore       image/crop/global latent/ROI latent 实体
Control State       assignment/agent status/budget/step/termination
```

### 4.3 Cost boundary

成本至少分开记录：

- Agent 调用；
- Tool 调用；
- model input/output token；
- Text 通信 token；
- Structure 序列化字节；
- Latent 实际加载字节；
- wall-clock latency；
- 错误、重试与无效动作。

## 5. Phase 0 — 契约与可观测性

### 5.1 Requirements

- 定义 Task、Assignment、Evidence、Claim、AnswerCandidate、Verdict；
- 定义 Budget、CostLedger、Event 与 stop reason；
- 定义 AgentObservation、AgentResult、CapabilityRequest；
- Phase 0 只冻结 Agent 输入/输出记录契约；WorkspaceDelta 与 Reducer 延后到 Phase 3，避免在单 Agent 阶段提前固化共享状态；
- JSON 可序列化，不把 image/tensor 直接嵌入领域记录；
- 事件日志能表示一次 episode 的顺序与成本；
- 提供确定性测试夹具和 seed；
- 为阶段、任务和预算提供配置入口。

### 5.2 Acceptance Criteria

- [x] 所有核心记录可 JSON round-trip；
- [x] 非法 confidence、负预算和悬空引用被拒绝；
- [x] CostLedger 能分别统计 Agent/Tool/token/structure/latent/latency；
- [x] Event log 能重放 step、actor、action、result 与 cost；
- [x] 同一测试 seed 产生相同 fixture；
- [x] 本地 pytest 全部通过。

本地 Phase 0 门禁已于 2026-08-21 通过；跨机器发布仍按 Phase 1 的 ARM Docker 门禁执行。

## 6. Phase 1 — 单 Generalist Agent

### 6.1 Requirements

- 实现一个通用 Generalist Agent，而不是立即拆分四个专家；
- 支持 image + question 输入；
- 支持至少 Crop、Inspect/Describe 和 Finish 工具/动作；
- 产生结构化 Evidence 与 AnswerCandidate；
- Generalist 的工具循环可使用 MockBackend 和真实 ModelBackend；
- 记录 model/tool/latency 成本与终止原因；
- Generalist 后续保留为论文 baseline。

### 6.2 Acceptance Criteria

- [x] Mock 图像任务完成 image -> tool -> evidence -> answer；
- [x] 至少一个真实图片 fixture 完成 CPU smoke；
- [x] 工具输出与答案具有 provenance；
- [x] 超预算、工具失败和格式错误能够终止并记录；
- [x] 相同 Agent 接口可替换 MockBackend 和真实后端；
- [x] 前置 ARM Docker 热挂载环境通过测试与 dry run；
- [x] 真实 VLM smoke 单独标记，只有实际模型运行后才勾选。

本地 Phase 1 严格门禁已于 2026-08-21 使用 Qwen3.8-27B Q4 通过 3/6 反事实图像对；同日前置 ARM64 服务器完成原生 CPU 镜像构建、热挂载哈希核验、容器测试以及 Phase 0/1 smoke。`lws2` 尚未挂载到前置机，因此不可变 release/image 的 PFS 发布仍待文件系统挂载后执行，不影响 Phase 2 本地开发。

## 7. Phase 2 — 单 Agent 内部演化

### 7.1 Requirements

- IntraAgentEvolution 是 Generalist/Expert 外部的可插拔 wrapper 或子图；
- 明确 Candidate、Variation、Fitness、Selection、Generation；
- 首版只使用一个简单、可复现的演化策略；
- discarded candidates 只存在 Private Scratchpad；
- 选中结果通过 AgentResult 提交；
- 记录 population、generation、fitness gain 和额外成本；
- 可以完全关闭演化以形成 baseline。

### 7.2 Acceptance Criteria

- [x] 日志中存在至少两个候选、一次 variation、fitness 和 selection；
- [x] 关闭演化后 Generalist 行为保持可运行；
- [ ] 相同预算下可比较 ReAct/retry/Best-of-N/evolution；
- [x] 未选中的候选不会进入 Shared Workspace；
- [x] 演化达到预算或无进展阈值后必定终止。

Phase 2 核心退出条件已于 2026-08-21 达成：可执行 runner 的真实事件顺序为 seed proposal/fitness → VLM variation → child fitness → selection → final Finish，关闭配置时复用 Phase 1 原路径；淘汰候选只留下 ID、哈希和分数，不进入公共 `AgentResult`/`RECORD_ADDED`。当前 fitness 是无标签的 evidence/confidence host evaluator，不宣称为 VLM judge。ReAct/retry/Best-of-N/evolution 的 matched-budget 横向比较属于 Phase 6 实验任务，不阻塞 Phase 3。

## 8. Phase 3 — EvoRS 特有表示层

### 8.1 Requirements

- 实现 episode-scoped SharedWorkspace；
- 所有更新通过 WorkspaceDelta + Reducer；
- 实现 Scene/Region/Group/Object 可跳级轻量层次；
- 只按问题创建相关节点，Group/Region 可选；
- 实现 Evidence/Claim/Answer 与 SpatialNode 的引用关系；
- 实现 ArtifactStore 与 VisualArtifact metadata；
- Workspace 只保存 artifact handle；
- 实现 global/ROI latent encoder 与 receiver adapter 接口；
- 先在单 Agent 中验证写入 -> scoped read -> 实际消费；
- latent channel feature flag 默认关闭。

### 8.1.1 本轮最小接口冻结

```text
WorkspaceState (episode scoped, immutable snapshot)
├── task
├── evidence / claims / answer_candidates
├── spatial_nodes
└── artifact_descriptors       # 只含 handle/metadata

WorkspaceDelta
├── episode_id / base_version / actor_id
└── append-only records

ArtifactStore                  # 保存 image/crop/latent 的物理数据
└── VisualArtifact             # Workspace 中对应的描述符
```

- Reducer 是唯一写入口，每次成功 apply 将 Workspace version 加一；
- `Scene` 是必需根节点；`Region`、`Group` 按问题需要创建，`Object` 可直接挂到 `Scene`、`Region` 或 `Group`；
- `SpatialNode` 只描述与当前问题相关的语义、bbox、层级、证据和 visual artifact 引用，不扩展成通用世界模型；
- latent descriptor 固定记录 encoder、layer、projector、shape、dtype、source ROI、compatible consumer、bytes 和内容哈希；
- 首个 encoder 使用真实像素计算的确定性 ROI feature 作为接口 smoke，明确标记为非论文视觉表征；学习型 vision encoder 的 ARM smoke 仍是 Phase 3 最终硬件验收项；
- latent consumer 必须通过 `ArtifactStore` 读取物理向量并返回实际读取字节数，Workspace 读取 descriptor 本身不算 latent 已消费；
- scoped view 在 Phase 3 只验证“单 Agent 被授予哪些记录就只能读哪些记录”，多 Agent inbox/Router/ACL 留到 Phase 4。

本阶段不实现 Controller、Router、Agent 通信、LangGraph 多 Agent 图、RL、持久化数据库或并发写入。

### 8.2 Acceptance Criteria

- [x] WorkspaceDelta 版本不匹配和非法引用被拒绝；
- [x] Object 可以直接挂 Scene 或 Region；
- [x] 不允许重复 ID、包含环和悬空 parent；
- [x] ArtifactStore 物理数据与 Workspace metadata 分离；
- [x] ROI latent descriptor 包含 encoder/layer/projection/shape/dtype/ROI/bytes；
- [x] 接收逻辑真实读取 latent，而不是只记录 handle；
- [x] latent 通信成本按实际加载字节计量；
- [x] 无 latent 时 Text + Structure 路径仍可运行；
- [ ] 前置 ARM 环境完成真实 encoder smoke 后才把 latent 标为完成。

Phase 3 表示层主线已于 2026-08-21 跑通：独立 runner 完成 `WorkspaceDelta → Reducer → scoped view → ArtifactStore latent consume`，实际读取 12 个 float32/48 bytes 并据此计费；关闭 latent 后结构路径仍可执行。当前 PixelStats encoder 是由真实 ROI 像素计算的机制验证器，`learned_visual_representation=false`，不会作为论文视觉 latent 报告。学习型 encoder 的 ARM smoke 保留为硬件验收项，但接口替换不阻塞 Phase 4 多 Agent 协作开发。

## 9. Phase 4 — 无 RL 的多 Agent 协作

### 9.1 Requirements

- 实现 Visual、Reasoning、Verification、Knowledge Expert；
- 实现 EvoRSController + RuleRoutingPolicy；
- Controller 只控制 Dispatch/Finish/Abort，不产生领域答案；
- 实现 ContextRouter、MessageEnvelope、Inbox 与 AccessGrant；
- ObservationBuilder 只构建 scoped view；
- 消息必须改变接收者下一步可见信息；
- Agent 只能提交 CapabilityRequest，不能直接指定并调用下一 Agent；
- LangGraph 运行时负责 conditional routing、reducer 和终止；
- 至少保留 fixed workflow、all-agent 和 rule routing baseline；
- 记录未来 RL 所需的真实 transition。

### 9.1.1 本轮最小协作闭环

```text
LangGraph START
  ↓
EvoRSController + RoutingPolicy
  ├── Dispatch(expert_id)
  ├── Finish(answer_candidate_ref)
  └── Abort(stop_reason)
  ↓
ObservationBuilder(scoped Workspace + recipient inbox grants)
  ↓
ONE Cognitive Expert
  ↓
WorkspaceDelta + CapabilityRequest(optional)
  ↓
WorkspaceReducer + ContextRouter(MessageEnvelope/AccessGrant)
  └────────────────────────────────────────────────────────→ Controller
```

- Cognitive Agent Pool 只包含 Visual、Reasoning、Verification、Knowledge；Controller/Policy 位于控制层；
- 首版 Expert 使用确定性规则实现机制闭环，后端协议保持可替换，真实 VLM 不作为 LangGraph 状态对象；
- Expert 不包含 `next_agent`，也不能直接调用另一个 Expert；下一 Agent 只能由 RoutingPolicy 选择；
- MessageEnvelope 只携带 recipient、modality、scope 与 Workspace refs，Inbox/AccessGrant 决定 ObservationBuilder 可披露的记录；
- 初始只有 Visual 获得任务级观察；每次 expert 提交后，Router 依据 Controller 的下一 dispatch 将新增 workspace refs 定向授权；
- `counting` 路由为 Visual → Reasoning → Verification，禁止 Knowledge；`classification/type` 路由为 Visual → Knowledge → Reasoning → Verification；
- fixed workflow、all-agent、rule routing 共享同一 runtime/budget，只替换 RoutingPolicy；
- transition 至少记录 step、policy、controller observation digest、decision、active expert、workspace version before/after、message IDs、cost、stop reason；
- LangGraph 固定使用 `langgraph>=1.2.11,<1.3` 的 `StateGraph/START/END/add_conditional_edges`，不依赖 LangChain；该版本支持 Python 3.10+；
- 首版单步只 dispatch 一个 Expert，不使用并行 fan-out，避免 reducer 并发合并语义提前复杂化。

本阶段不实现 learned routing、RL、Agent 自由互调、全 Workspace 自动共享或论文性能结论。

### 9.2 Acceptance Criteria

- [x] “统计飞机”任务不会激活 Knowledge Expert；
- [x] “判断机场类型”可激活 Visual + Reasoning + Knowledge；
- [x] 删除一条消息后目标 AgentObservation 发生变化；
- [x] 非接收者不能自动读取未授权记录；
- [x] Controller 不能 Finish 一个不存在的 AnswerCandidate；
- [x] 预算耗尽、无进展、工具失败和正常 Finish 均有明确 stop reason；
- [x] 固定、全 Agent、规则路由在同一预算配置下可运行；
- [x] 每个 episode 可 replay Agent 路径、披露内容和成本；
- [ ] 前置 ARM Docker 热挂载环境通过完整 smoke/test。

Phase 4 本地主线已于 2026-08-21 跑通：LangGraph `StateGraph` 负责 `Controller → conditional dispatch → one Expert → Workspace reducer → Controller` 循环；2 个 scripted task × fixed/all/rule 共 6 次运行全部完成，transition 可重放 Agent 路径、定向披露、Workspace version、成本及最终答案引用。规则 counting 路径不含 Knowledge，机场类型路径包含 Visual/Knowledge/Reasoning/Verification；将 `max_agent_calls` 降为 2 会真实截断轨迹并以 `budget_exhausted` 结束。该结果只证明协作机制，不是模型性能实验。

## 10. Implementation Order

```text
0.1 schemas + validation
0.2 budget/cost/events/config
0.3 deterministic fixtures/tests

1.1 tool protocol + crop/inspect tools
1.2 GeneralistAgent
1.3 mock end-to-end runner
1.4 real backend adapter/smoke

2.1 candidate/evolution protocol
2.2 deterministic variation + fitness + selection
2.3 Generalist evolution runner and baselines

3.1 workspace + delta + reducer
3.2 spatial hierarchy
3.3 artifact store
3.4 vision encoder + latent consumer
3.5 scoped observation

4.1 expert roles
4.2 controller/routing policy
4.3 message/router/inbox/access
4.4 LangGraph graph
4.5 fixed/all/rule baselines and replay
```

## 11. Verification

### Local

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

每个 phase 还必须有独立 dry-run，不只依赖 unit tests。

### 前置 ARM 服务器

目标约束：

- Ubuntu 24.04 ARM64；
- 4 vCPU / 16 GiB；
- Docker 环境；
- 源码只通过 bind mount / PFS 热挂载；
- image 与 code 可放置于 `lws2`；
- 不把密码、OBS key 或其他凭证写入仓库；
- 不在验证主机上启动 Ascend 训练。

远端验证前必须先确认 SSH host key。当前探测到的 ED25519 指纹与本机历史记录不一致，未确认前禁止绕过校验。

建议远端验收命令最终统一为：

```text
docker compose --env-file configs/docker/compose.defaults.env build cpu
docker compose --env-file configs/docker/compose.defaults.env run --rm cpu
docker compose --env-file configs/docker/compose.defaults.env run --rm cpu python3 scripts/<phase>_dry_run.py
```

实际 PFS 根目录、镜像 tar 路径和代码 checkout 路径必须在只读盘点后再固定，不能根据名称猜测后执行写操作。

## 12. Open Questions

- 前置 ARM 服务器或 `lws2` 中是否已有可用于真实 VLM/encoder smoke 的模型与 ARM 镜像；
- Phase 1 首个真实 VLM 的规模和加载方式；
- Phase 2 首个 candidate 类型与 role-specific fitness；
- Phase 3 首个真实 vision encoder/layer/projector；
- Phase 4 Knowledge Expert 使用静态知识、RAG 还是工具；
- LangGraph 与 Python/verl 的最终兼容版本；
- PFS 上镜像与代码的规范目录。

这些问题不阻塞 Phase 0 和 Mock 驱动的 Phase 1 实现，但会阻塞相应 phase 的“真实模型/真实 latent”完成标记。
