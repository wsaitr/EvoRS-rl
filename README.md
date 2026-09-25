# EvoRS-Comm

面向遥感视觉理解的自然演化多智能体协作研究原型。当前主架构是：

```text
单 Generalist Agent + Agent 内部演化
  -> EvoRS Workspace + 按需层次描述 + ArtifactStore/latent
  -> Cognitive Expert Pool + Evolution Controller + Router
  -> 强化学习集中式动态路由
```

Agent 内部使用 candidate、variation、fitness、selection 进行演化；多 Agent
之间不称为“演化”，而由控制层进行动态路由和策略学习。Agent 不能直接调用另一个
Agent，信息通过受控 Workspace 引用和显式 MessageEnvelope 披露。

## 当前完成状态

| 阶段 | 状态 | 主要证据 |
|---|---|---|
| Phase 0 契约与复现基础 | 完成 | 配置、manifest、预算、成本、事件与回放测试 |
| Phase 1 单 Agent 闭环 | 完成 | 真实 Qwen VLM 感知门禁 |
| Phase 2 单 Agent 内部演化 | 完成 | candidate/variation/fitness/selection 私有轨迹 |
| Phase 3 EvoRS 表示层 | 完成 | Workspace、稀疏层次节点、ArtifactStore、latent 消费测试 |
| Phase 4 无 RL 多 Agent | 完成 | 20 样本 paired gate：Direct 50%、Single 60%、Multi 70% |
| Phase 5 RL 动态路由 | 完成机制门禁 | 720 个训练轨迹、6 个留出样本、冻结专家、动作-轨迹因果审计与 checkpoint |
| Phase 6 论文级实验 | 未开始 | 需要完整 baseline、消融、多 seed 和成本曲线 |

Phase 4 的 20 样本结果和 Phase 5 的确定性 fixture 都是开发门禁，不是论文级统计结论。

## 快速验证

```bash
python -m pip install -e ".[dev]"
python -m pytest -q

python scripts/validate_phase5.py \
  --evidence evidence/2026-09-03_phase5_routing

python scripts/audit_phase0_5.py \
  --output evidence/2026-09-03_phase0_5_completion_audit \
  --run-tests
```

## 训练 Phase 5 Controller

```bash
python scripts/train_phase5_routing.py \
  --config configs/experiments/phase5_routing.yaml
```

该训练只更新 7×21 的离散 `LearnedRoutingPolicy`，不更新 VLM 或 Cognitive
Expert。输出包含 `checkpoint.json`、逐 episode 事件、奖励分解、审计与 checksums。

将 checkpoint 用于真实遥感多 Agent 推理：

```bash
python scripts/run_real_multi_agent.py \
  --manifest configs/experiments/vrsbench20_manifest.jsonl \
  --model-config configs/models/qwen38_27b_q4_local.yaml \
  --routing-checkpoint evidence/2026-09-03_phase5_routing/checkpoint.json \
  --output-dir outputs/learned-routing-smoke \
  --limit 1
```

## Ascend / lws2

当前离散 Controller 在 CPU 训练，以节省 NPU 时间。lws2 仍保存相同代码、配置、
checkpoint 和证据；4×Ascend 留给后续 token-policy/VLM 训练和 Phase 6。

```bash
python scripts/train_npu.py verify \
  --checkpoint evidence/2026-09-03_phase5_routing/checkpoint.json \
  --output evidence/ascend-environment.json \
  --require-npu --require-verl

docker compose --env-file configs/docker/compose.defaults.env \
  -f docker-compose.pfs.yml run --rm train-phase5
```

本地完成审计见 `evidence/2026-09-03_phase0_5_completion_audit/`。服务器实机部署
必须另外保留 SSH、镜像、挂载和运行日志，不能用本地 compose 配置校验替代。

完整部署见 [DEPLOY.md](DEPLOY.md)，当前架构与验收边界见
[EVORS_COMM_ARCHITECTURE_V2.md](doc/论文/1指导书/EVORS_COMM_ARCHITECTURE_V2.md)
和 [EVORS_PHASE5_RL_SPEC.md](doc/论文/1指导书/EVORS_PHASE5_RL_SPEC.md)。
