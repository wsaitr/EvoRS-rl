# EvoRS Phase 5：强化学习动态路由实现规格

## 1. 目标

Phase 5 只训练控制层的 `LearnedRoutingPolicy`，冻结 Visual、Knowledge、
Reasoning、Verification 四类 Cognitive Expert。学习到的动作必须在专家执行前产生，
并真实决定后续 Agent 激活、信息披露方式和终止行为。

本阶段的正式表述是“基于强化学习的集中式动态协作优化”，不是多 Agent 间演化，
也不在尚未完成多策略联合训练时宣称实现了 MAGRPO 或新的 HM-MAGRPO 算法。

## 2. 已发现的旧实现问题

2026-09-02 的代码审计确认，旧 `rl/hmagrpo.py`、`train_stage2.py` 与
`train_npu.py --verl` 不能作为 Phase 5 证据：

- 只计算标量 loss，没有可训练参数、优化器或参数更新；
- 通信动作在旧 orchestrator 完成 episode 后补采样，不影响该 episode；
- task score 使用“是否停止”而不是真实答案正确性；
- `--verl` 只生成配置和 JSONL，没有启动训练；
- 使用已降级为 baseline 的 Global/Local/Hierarchy/Residual Agent 主链。

这些文件必须被真实的 Controller-RL 主链替代或明确降级，不能继续以“完整训练”命名。

## 3. 设计决策

### 3.1 策略形式

首版采用独立的小型离散策略网络，而不是让 VLM 生成路由 token：

- 输入：`ControllerObservation` 的固定版本特征向量；
- 输出：带有效动作 mask 的离散结构化动作；
- 训练：on-policy policy gradient（带移动基线、熵正则和梯度裁剪）；
- 推理：纯 Python 读取 JSON checkpoint，不要求加载训练框架或 VLM 权重；
- 专家：全部冻结，只作为环境动力学执行。

该选择符合架构文档 10.4 节的边界：对独立小型离散网络，直接 policy-gradient
实现比把动作伪装成 LLM token 再接 verl 更直接。保留 verl Agent Loop/Reward 数据适配，
供以后切换 token policy；当前 Phase 5 的真实参数更新由离散 Controller trainer 完成。

### 3.2 动作空间

```text
Dispatch Visual
Dispatch Knowledge
Dispatch Reasoning(Text+Structure)
Dispatch Reasoning(Text+Structure+Latent)
Dispatch Verification
Finish(verified candidate)
Abort(no progress)
```

`Reasoning` 的两个动作使用不同 `ContextSpec`，因此会产生不同的通信成本和可见信息。
动作 mask 至少约束：

- 无视觉证据时不能 Reasoning/Knowledge/Verification/Finish；
- 无 proposed candidate 时不能 Verification；
- 无 verified candidate 时不能 Finish；
- 无兼容 latent artifact 时不能选择 latent 动作；
- 不允许超过最大步数、总预算或明显无进展的重复激活。

### 3.3 环境与奖励

`LangGraphMultiAgentRuntime` 是真实环境执行器。策略每次 `decide` 时保存采样记录，
Runtime 随后执行该 `Dispatch | Finish | Abort`，原有 transition/message/workspace 日志
是轨迹事实源。

奖励由环境侧 `RoutingRewardManager` 在 episode 结束后计算，reference answer 不进入
Controller 或任何 Expert observation：

```text
R = task_score
  + evidence_quality_gain
  + structure_quality_gain
  - message_cost
  - agent_cost
  - tool_cost
  - redundancy_cost
  - failure_or_invalid_penalty
```

结构和证据项按 episode 的最终增量计算一次，禁止每步重复奖励同一静态状态。

## 4. 功能要求

1. `LearnedRoutingPolicy` 必须使用与 `RuleRoutingPolicy` 相同的 `RoutingPolicy` 接口。
2. checkpoint 必须包含 schema、特征版本、动作表、参数、训练步数和校验摘要。
3. 训练 episode 必须记录 state feature、action mask、采样 action、采样概率、reward 和更新量。
4. 训练前后策略参数 SHA-256 必须不同；零更新或 NaN 必须使训练失败。
5. `MultiAgentTransition.decision` 必须可追溯到该步的已采样 action。
6. 专家冻结指纹在训练前后必须一致。
7. 数据必须显式划分 train/eval；eval 禁止参数更新和探索采样。
8. 训练证据必须包含配置、逐 episode 事件、checkpoint、汇总、审计和 checksums。
9. verl 适配必须输出官方 RLHF 数据字段、结构化 action trace 和自定义 reward 输入；
   dry-run 必须能在未安装 verl 的开发机上验证契约，不能谎称已执行 verl 训练。
10. Ascend/lws2 入口必须复用同一 checkpoint、manifest 和证据目录，不另建旧 Agent 主链。

## 5. 验收标准

- **AC-1 真实动作时序**：测试证明 action 在对应 Workspace version 改变前采样；删除或
  强制替换 action 会改变后续 `agent_path`、Workspace version 或通信成本。
- **AC-2 真实学习**：至少一次可复现训练产生非零 optimizer update，训练前后参数摘要不同，
  保存后的 checkpoint 可重新加载并复现 greedy 决策。
- **AC-3 冻结边界**：训练前后四个 Expert 的冻结指纹完全相同；训练器只持有策略参数。
- **AC-4 标签隔离**：reference answer 只出现在环境 sample/reward/evidence 文件中，
  不出现在 Controller/Expert observation 序列化结果中。
- **AC-5 动作有效性**：所有无效 Finish、latent、Verification 和重复动作都被 mask 或拒绝。
- **AC-6 奖励可审计**：总奖励严格等于各分量之和，成本分量来自真实 `CostVector` 与消息。
- **AC-7 轨迹闭环**：训练与 greedy eval 均通过 `MultiAgentRunResult.replay()`。
- **AC-8 机制门禁**：在冻结的确定性训练 fixture 上，学习策略相对未训练策略改善平均奖励，
  且至少两类任务产生不同协作路径、至少两类任务产生不同 Reasoning 通信模态；Hybrid
  必须披露显式 latent ref 并产生非零 latent 成本。该门禁只证明机制，不作为论文精度结论。
- **AC-9 工程回归**：全量离线测试通过；Phase 0–4 的既有证据与运行入口不被破坏。
- **AC-10 可迁移性**：Ascend dry-run 能验证依赖、配置、数据和 checkpoint，并生成唯一启动命令；
  若没有可访问的 4×Ascend 主机，不能把 dry-run 写成已完成 NPU 训练。

## 6. 实现任务

1. 增加路由动作 codec、固定特征 encoder、mask 和 JSON checkpoint。
2. 将 `LearnedRoutingPolicy` 替换为真正可训练策略，保留未训练时的安全规则回退。
3. 增加奖励分解、on-policy trainer、冻结指纹和 episode audit。
4. 增加 `EvoRSAgentLoop`/verl record 与 reward adapter。
5. 重写 Phase 5 CLI、本地机制 fixture、Ascend/lws2 启动入口。
6. 新增测试并生成一次真实本地训练/eval 证据。
7. 更新总架构文档的 Phase 5 状态，执行 Phase 0–5 完成审计。

## 7. 验证命令

最终命令以实现后的 CLI 为准，至少包括：

```text
python -m pytest tests/test_routing_rl_*.py -q
python scripts/train_phase5_routing.py --config configs/experiments/phase5_routing.yaml
python scripts/validate_phase5.py --evidence <phase5-evidence-dir>
python -m pytest -q
```

## 8. 非目标

- Phase 6 的多 benchmark、多 seed 论文结论和显著性检验；
- 联合更新 VLM/Expert 参数；
- 把单 Controller 训练宣传成严格的 multi-policy MARL；
- 在没有真实运行日志时宣称完成 verl 或 Ascend 训练；
- 新算法命名、理论收敛声明或 HM-MAGRPO 论文贡献声明。
