# 给 Coding Agent / Claude Code / Codex 的实现提示词

你要实现一个研究代码库：

**Remote-Sensing Hierarchical Multimodal Communication for Multi-Agent Reinforcement Learning**

不要一次性假装所有模块都完成。必须逐阶段执行、运行测试、报告失败。

## 1. 论文目标

四个贡献：
1. 遥感多 Agent 自然演化协作框架；
2. `Scene -> Region -> Group -> Object` 分层多模态通信；
3. 支持 recipient / level / modality / payload 的 multi-agent RL；
4. 在遥感 benchmark 上提升任务分数和通信效率。

Agent 通信不只允许文本：
- text
- structured subtree / node reference
- optional visual latent handle

## 2. 上游参考

CoMLRL / MAGRPO：
- https://github.com/OpenMLRL/CoMLRL
- https://ojs.aaai.org/index.php/AAAI/article/view/40487

MARTI：
- https://github.com/TsinghuaC3I/MARTI
- https://openreview.net/forum?id=E7jZqo0A50

策略：
- Stage 1 不依赖它们；
- Stage 2 先接 CoMLRL/MAGRPO；
- MARTI 是第二实现或大规模异构训练候选；
- 不直接魔改 third_party，写 adapter；
- 正式实验前 pin commit。

## 3. 数据集

优先：
1. VRSBench：训练/开发
2. CHOICE：外部 evaluation
3. XLRS-Bench-lite/full：超大图与效率
4. GEO-Bench-VLM：广覆盖外部泛化

禁止自行补造 CRS/GSD/sensor metadata。

## 4. Stage 1

### 4.1 SceneTree

实现：
```python
SceneNode
SceneTree
```

Node：
```python
id
level: scene | region | group | object
semantic
bbox
confidence
status: proposed | verified | rejected
parent_id
evidence_refs
latent_handle
attributes
```

Tree 操作：
- add/update/get
- children/subtree
- query_by_level
- query_by_semantic
- verify/reject
- serialize/replay

### 4.2 MessageBus

Message：
```python
sender
receiver
spatial_level
modalities
text
node_ids
bbox_refs
latent_handles
metadata
```

必须统计：
- message count
- text token/char
- struct nodes
- bbox refs
- latent bytes

### 4.3 Agents

实现：
- GlobalAgent
- LocalAgent
- HierarchyAgent
- VerifierAgent
- ResidualAgent

不要把 Agent 业务写死在 orchestrator。

### 4.4 Controller

先 RuleController：
输入 question/tree/history/budget；
输出 next_agent/target_node/modality/stop。

### 4.5 Backend

统一：
```python
generate(image, prompt, structured_context=None)
encode_roi(image, bbox)
```

实现：
- MockBackend
- TransformersVLMBackend
- 可选 VLLMBackend

### 4.6 Hardware

统一 DeviceSpec：
- auto/cuda/npu/cpu

禁止 `.cuda()`。
Ascend 只在 backend 层 import torch_npu。
分布式：
- CUDA -> NCCL
- NPU -> HCCL

### 4.7 Stage 1 验收

先运行：
```bash
pytest -q
python scripts/dry_run_stage1.py
```

通过后才接真实模型。

## 5. Stage 2 RL

### 5.1 CommunicationAction

```python
CommunicationAction(
    task_action,
    recipient,
    spatial_level,
    modality,
    payload_node_ids
)
```

### 5.2 Reward

第一版：
```text
R_task - λ_cost*C_comm - λ_dup*C_dup
```

收敛后加：
- R_structure
- R_evidence
- R_message_value

### 5.3 CoMLRL Adapter

不要假设 CoMLRL 原生支持 VLM structured message。
实现：
```python
CoMLRLTrajectoryAdapter
```

读取 pin 后的真实 upstream API，再做转换。

如果 upstream trainer 只接受文本：
1. 先跑 text-only MAGRPO；
2. tree context 作为 baseline；
3. structured communication action 用专门 action token/side metadata；
4. 最终实现 communication policy head；
5. 不要偷偷丢掉 struct payload。

### 5.4 HM-MAGRPO

比较：
- vanilla text MAGRPO
- MAGRPO + tree context
- HM-MAGRPO structured action
- HM-MAGRPO + latent（可选）

## 6. Latent channel

必须 feature flag：
```yaml
latent_channel:
  enabled: false
```

先同 backbone。
不要一开始做异构 latent 对齐。

真实 latent 必须是 tensor/hidden state，不允许拿 bbox 字符串冒充。

## 7. Stage 3

Baseline：
- single VLM
- CoT
- fixed MAS
- text-only MAS
- vanilla MAGRPO
- structured no-RL
- HM-MAGRPO
- + latent

记录：
- benchmark score
- agent calls
- messages
- text tokens
- struct nodes
- latent bytes
- crops
- latency
- peak memory

核心实验至少 3 seeds。

## 8. 工程规范

必须：
- type hints
- dataclass
- unit tests
- config driven
- deterministic seed
- JSONL trajectories
- structured logging
- 明确异常

禁止：
- 每次把整个 tree stringify 后塞 prompt 就称“结构通信”
- latent handle 冒充 latent tensor
- test answer 泄漏进 trajectory
- 没实验就写“提升 xx%”
- 把 CUDA/Ascend 逻辑散落在 Agent 层

## 9. 执行顺序

每一步都实际执行测试并汇报：

1. SceneTree
2. MessageBus
3. Mock agents + orchestrator
4. VRSBench adapter
5. Real VLM 10 samples
6. Rule dynamic collaboration 100 samples
7. CUDA smoke
8. Ascend smoke
9. CoMLRL text MAGRPO
10. structured communication policy
11. HM-MAGRPO pilot
12. full experiments

遇到 upstream API 不符：
**读真实源码再改 adapter，不准猜接口。**
