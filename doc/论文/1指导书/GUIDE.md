# RS-HMComm 三阶段实施计划与指导书

## 0. 研究目标

论文四个贡献对应四个工程目标：

1. **宏观框架：遥感多 Agent 自然演化协作框架**
   - 通信对象、空间层级、消息模态和协作路径不完全写死。
2. **方法创新 A：分层多模态通信**
   - 共享结构：`Scene -> Region -> Group -> Object`
   - 消息：`Text + Structured State + optional Visual Latent`
3. **方法创新 B：Structured-Communication MARL**
   - 把 recipient / level / modality / payload 纳入 communication policy。
4. **经验贡献：性能和通信效率**
   - 在相同预算下获得更高任务分数，或相同分数下用更少通信。

---

# 1. 数据集选择

## 1.1 VRSBench：主训练 + Stage-1 调试

建议作为整个项目的主训练集。

适合原因：
- 29,614 张遥感图；
- 29,614 条 detailed caption；
- 52,472 条 object referring；
- 123,221 个 VQA；
- 构建数据时包含 object category、bbox、position、size 等对象信息；
- VQA、grounding、caption 同一体系，天然适合构建和验证层次共享状态。

推荐规模：
- smoke：100–500 samples
- dev：2k–5k QA
- RL pilot：5k–20k QA
- full：机制稳定以后再上

注意：部分图像来自 DOTA，正式实验遵守对应学术许可。

## 1.2 CHOICE：主要外部 evaluation

默认**不拿 CHOICE 训练**，避免 benchmark contamination。

特点：
- 10,507 个问题；
- perception + reasoning 两大维度；
- 6 个 L2 维度、23 个 L3 任务；
- MCQ 有明确答案；
- 包含 object counting、localization、spatial relationship、change detection 等。

用法：
- Stage 1：官方 460-sample subset 做兼容性 smoke test；
- Stage 3：完整 CHOICE 做外部泛化。

## 1.3 XLRS-Bench：超大图与通信效率压力测试

特点：
- 平均图像约 8500×8500；
- 16 个任务；
- 45,942 条标注；
- counting / regional reasoning / grounding 很适合验证分区域协作。

这里重点看：
- Global Agent 是否只产生高价值 Region；
- Local Agent 是否只看必要 crop；
- Structured Message 是否能直接携带 bbox / subtree；
- 是否减少重复整图编码。

先跑 XLRS-Bench-lite，后跑 full。

## 1.4 GEO-Bench-VLM：广覆盖外部测试

31 个 fine-grained task、8 类大任务，包括：
scene/object classification、counting、localization、segmentation、captioning、
event detection、non-optical、temporal understanding。

用途：证明方法不是只对 VRSBench 有效。

---

# 2. CoMLRL 还是 MARTI？

## 2.1 推荐顺序

### Stage 1
两个都不绑死。
先把自己的 inference + shared state + message bus 跑通。

### Stage 2 第一版
**优先 CoMLRL / MAGRPO。**

原因：
- 直接针对 fully cooperative multi-LLM RL；
- 有 MAGRPO、MAREINFORCE、MARLOO、MAREMAX；
- 相比 MARTI 更轻；
- 更适合先改 action/reward 验证“结构通信能不能 work”。

### Stage 2 第二版 / 大规模
**再接 MARTI。**

MARTI 优势：
- graph workflow；
- heterogeneous agents；
- async tool use；
- centralized interaction + distributed policy training；
- 多种 RL；
- MARTI-v2 还有 multi-agent tree search。

但官方很多训练示例是约 **8×80GB GPU / agent** 的级别，
不适合拿来做第一轮通信协议 debug。

---

# 3. 当前最大风险：现有 MARL 框架主要面向文本 LLM

我们需要的是：
- image input
- tree state
- bbox
- crop reference
- optional visual latent tensor

因此必须分两级做。

## 3.1 MVP：Text + Structure + Image Reference

统一消息：

```text
Message {
    sender
    receiver
    level
    modalities
    text
    node_ids / subtree
    bbox_refs
    latent_handles(optional)
}
```

需要视觉信息时，接收 Agent 根据 bbox 从原图重新 crop。

这是跨 NVIDIA / Ascend 最稳的第一版。

## 3.2 Experimental：真正 Visual Latent

共享 Vision Encoder 对 ROI 得到：

```text
latent_handle -> Tensor[K, D]
```

然后接收 Agent 拿到：
```text
(node metadata, visual latent)
```

困难：
1. 同 backbone 比较容易；
2. 异构 VLM hidden dim 不同，需要 projector；
3. vLLM API 不一定暴露/注入中间 vision hidden states；
4. Direct Transformers 更适合 latent prototype；
5. Ascend/NVIDIA 都要分别测 dtype、算子、显存/内存生命周期。

**结论：latent 是增强实验，不阻塞 MVP。**

---

# 4. Stage 1：动态协作与通信推理框架

## 4.1 共享树

输入：
```text
image + question
```

共享：
```text
Scene
└─ Region
   └─ Group
      └─ Object
```

Node 至少：
```text
id
level
semantic
bbox
confidence
status
parent_id
evidence_refs
latent_handle(optional)
attributes
```

## 4.2 Agent

第一版：
- GlobalAgent：整图 -> Region
- LocalAgent：Region -> Group/Object
- HierarchyAgent：层次聚合与关系
- VerifierAgent：证据检查、状态更新
- ResidualAgent：先保留接口，后面 RL 再让它自然形成补充能力

## 4.3 MessageBus

所有 Agent 间通信都必须经过 MessageBus。

必须记录：
- sender/receiver
- text
- node_ids
- bbox
- modalities
- latent handles
- 时间
- message count
- text token/char
- struct node 数
- latent bytes（后续）

否则无法做通信效率实验。

## 4.4 动态协作

第一版先 RuleController：

- 没 Region -> Global
- counting/localization 且缺细粒度节点 -> Local
- 有局部证据但层次不完整 -> Hierarchy
- confidence 低/冲突 -> Verifier
- 多轮失败 -> Residual
- evidence sufficient -> Answer/Stop

目的：先证明“共享显式状态能驱动协作”。

## 4.5 Stage 1 验收

必须：
- [ ] mock episode 跑通
- [ ] 真 VLM 跑通 VRSBench sample
- [ ] Global 写 Region
- [ ] Local 读取 Region bbox 后写 Group/Object
- [ ] Hierarchy 读取 subtree
- [ ] Verifier 能修改 status/confidence
- [ ] MessageBus 可 replay
- [ ] CUDA 真实推理 smoke
- [ ] Ascend 真实推理 smoke

Stage 1 不要求 SOTA，只要求**可训练、可回放、可统计**。

---

# 5. Stage 2：Structured-Communication MARL

## 5.1 Action

建议：

```text
a_i^t = (
    task_action,
    recipient,
    spatial_level,
    message_modality,
    payload_selection
)
```

其中：
- task_action：inspect / verify / aggregate / answer / stop
- recipient：哪个 Agent
- spatial_level：Scene / Region / Group / Object
- modality：text / struct / latent / combinations
- payload：发送哪些 node/subtree/latent

## 5.2 Reward

第一版只开：

```text
R = R_task
  - λ_cost * C_communication
  - λ_dup  * C_redundancy
```

确认能收敛后，再加入：
- R_structure
- R_evidence
- R_message_value

不要第一次把 reward 搞得过于复杂。

## 5.3 MAGRPO 改造

每步 rollout 必须保留：

```text
state_t
agent_id
task_action
recipient
level
modality
payload
logprob
team_reward
communication_cost
```

实验顺序：

1. Vanilla MAGRPO：text only
2. MAGRPO + tree 作为 context
3. HM-MAGRPO：communication action 进入 policy
4. HM-MAGRPO + latent（可选）

## 5.4 小规模 Pilot

先小后大：
- 2–3 Agent
- 500–2000 train samples
- 每 episode 3–5 次通信
- 小模型或 7B + LoRA
- 3 seeds

看 signal：
- reward 是否上升；
- 是否超过 rule controller；
- communication cost 是否下降；
- policy 是否按任务选择不同 level/modality。

## 5.5 Stage 2 验收

- [ ] rollout 完整
- [ ] loss/reward 无 NaN
- [ ] 至少一类任务优于未训练 controller
- [ ] 不坍缩成“永远全部 Agent”
- [ ] 不坍缩成“永远不通信”
- [ ] structured payload 真被 receiver 使用
- [ ] 三个 seed 趋势一致

---

# 6. Stage 3：完整实验

训练：
- VRSBench train

测试：
- VRSBench held-out
- CHOICE
- XLRS-Bench
- GEO-Bench-VLM

Baseline：
- Single VLM
- Single + CoT
- Fixed MAS
- Text-only MAS
- Vanilla MAGRPO
- Structured communication without RL
- HM-MAGRPO
- HM-MAGRPO + latent（若实现成功）

消融：
- text only
- structure only
- text + structure
- text + structure + latent

层级消融：
- flat object list
- Scene/Region/Object
- Scene/Region/Group/Object

协作消融：
- fixed workflow
- rule controller
- learned controller
- MARL communication

效率指标：
- Avg Agent Calls
- Communication Turns
- Text Tokens
- Structured Nodes
- Latent Bytes
- Image Crops
- Latency
- Peak Memory

必须画：
**Score vs Communication Cost**。

---

# 7. NVIDIA + 华为 Ascend 双后端

原则：
业务代码禁止 `.cuda()`。

统一：
```python
device = resolve_device()
tensor = tensor.to(device)
```

## NVIDIA
- PyTorch CUDA
- NCCL
- Transformers 直连做 latent
- vLLM 做高吞吐 inference/rollout

## Ascend
- CANN + torch_npu
- HCCL
- vllm-ascend 做高吞吐 inference
- Transformers + torch_npu 做中间 hidden state prototype

Ascend 的 torch / torch_npu / CANN / vllm-ascend 版本必须按官方兼容矩阵安装，
不要在通用 requirements 里瞎 pin。

**不承诺 CoMLRL/MARTI 原仓库不改就能在 Ascend 一键训练。**
正确做法：
1. shared state/protocol 与硬件解耦；
2. CUDA 先验证算法；
3. Ascend 先验证 inference + rollout；
4. 再逐项替换第三方 CUDA-specific 组件。

---

# 8. 建议时间

## Stage 1：1–2 周
Day 1–2：dataset adapter + SceneTree + MessageBus + mock  
Day 3–5：VLM backend + 4 固定 Agent  
Day 6–8：rule dynamic collaboration + 100 sample  
Day 9–10：CUDA / Ascend smoke + logging/replay

## Stage 2：2–4 周
Week 1：CoMLRL text MAGRPO baseline  
Week 2：structured action + HM-MAGRPO  
Week 3：reward/3 seeds/LoRA  
Week 4：latent prototype / MARTI optional

## Stage 3：2–4 周
full VRSBench + CHOICE + XLRS + GEO-Bench-VLM  
主表 + 消融 + 效率 + failure cases + qualitative trajectories

---

# 9. 现在立刻做的 5 件事

1. `python scripts/dry_run_stage1.py`
2. VRSBench 100 条替换 mock
3. 接真实 Qwen2.5-VL backend
4. 检查 tree/message log 是否真的被 Agent 使用
5. 稳定以后再 clone CoMLRL 做 Stage 2

不要先上 MARTI-v2，不要先做 5 个 7B Agent，不要先做 latent。
