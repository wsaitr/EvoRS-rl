# EvoRS-Comm 阶段工作报告

日期：2026-10-08

代码基线：`530914e2d39847de29249cecf0dd1a663cea5f1c`

协作分支：[`codex/phase0-5-modelarts`](https://github.com/wsaitr/EvoRS-rl/tree/codex/phase0-5-modelarts)

本报告总结已有代码与保存的实验记录。本次工作是文档整理，没有重新运行模型、训练任务或云端训练作业。实验发生时间以对应证据目录为准。

## 1. 我们做到了哪里

项目已经从设计讨论发展为可运行的遥感 Agent 研究原型：先让单 Agent 调用视觉工具、重新观察图像，再加入私有候选演化、共享状态与空间描述，最后实现认知专家协作和可训练的路由控制器。代码、实验记录、配置和部署材料已经上传协作分支，`main` 尚未合并这些成果。

目前最具体的结果是：在同一组 20 个开发样本上，Direct VLM、单 Agent、多 Agent 分别答对 10、12、14 题。单 Agent 超过单次 VLM 的开发目标已在这组样本上达到，但调用成本同时增加。另一个独立的确定性环境实验验证了路由策略可以学习不同任务使用不同专家和通信模态。完整论文实验尚待执行。

| 阶段 | 已实现的内容 | 当前证据与边界 |
|---|---|---|
| Phase 0 | 数据 manifest、任务与动作契约、预算、分项成本、事件日志 | 为样本重放和成本审计提供基础 |
| Phase 1 | Generalist Agent、真实 VLM 接口、视觉工具、证据到答案闭环 | 保存真实模型感知门禁与 20 样本对照 |
| Phase 2 | candidate、variation、fitness、selection 私有循环 | 保存候选谱系与选择轨迹；独立性能贡献待消融 |
| Phase 3 | Workspace、稀疏层次描述、ArtifactStore、视觉 latent | 已有实体读取与授权计费证据；视觉 token 融合待实现 |
| Phase 4 | 专家池、Controller、Router、LangGraph runtime | 保存三臂 20 样本开发对照及真实多专家运行 |
| Phase 5 | 集中式 RL 路由、动作掩码、checkpoint、轨迹审计 | 18 个确定性训练任务、720 条训练轨迹、6 个留出任务；有真实 VLM 接入 smoke |
| Phase 6 | 正式评测方案已整理 | 完整 baseline、消融、多 seed、成本曲线尚未完成 |

## 2. 为什么先做单 Agent

研究目标是提升 VLM 对遥感图像的感知与推理能力。遥感任务中的小目标、密集重复目标和大尺度背景，使一次整图问答容易漏看。Agent 必须能够改变观察方式，把图像或裁剪结果真正送回 VLM，才有机会改善答案。

因此，单 Agent 主线围绕主动感知构建：从整图观察出发，按需要使用 crop、网格裁剪、联系图和空间拼接等工具，让 VLM 看到新的视觉输入，再输出有来源记录的答案。工具能力在 `src/rs_hmcomm/tools/`，主动感知在 `agents/active_perception.py`，整合表示层与演化的运行入口在 `runtime/integrated_single_agent.py`。

工程保留了不同用途的入口：最小 Generalist、用于开发对照的主动感知 Agent、带表示层与内部演化的完整单 Agent。20 样本中的 Single 结果来自主动感知对照，不能直接归因为完整演化或 latent 的收益。

## 3. “演化”和“多 Agent 协作”如何分工

### 3.1 单 Agent 内部演化

演化发生在单个专家的私有执行过程中。系统维护候选、父子关系、变异来源和适应度分量，通过选择决定保留哪个候选。未入选草稿不自动写入公共 Workspace，从而避免其他专家把中间猜测当作已提交证据。

当前整合运行器的评价器使用多视图证据支持、已有主动感知选择、对象计数一致性和来源成本等可计算信号；评价时没有参考答案。需要注意，单 Agent 的 `latent_consistency` 项目前是答案计数与已消费 ROI latent 数量的一致性检查，并不是已学习的视觉语义判别器。完整的演化价值仍需与相同调用预算的重复采样、重试和 Best-of-N 对照。

主要实现：`evolution/engine.py`、`evolution/protocol.py`、`runtime/integrated_single_agent.py`。

### 3.2 多 Agent 动态协作

专家按认知职责划分为 Visual、Knowledge、Reasoning 和 Verification。Evolution Controller/Policy 位于控制层，读取紧凑状态、选择下一位专家、决定上下文模态及终止。

专家之间通过共享记录与路由消息交接。Visual 提交观察证据、节点及制品描述；Reasoning 提出答案候选；Verification 提交验证结果；Controller 按状态和预算控制执行与最终选择。多个角色可以复用同一个 VLM 服务，各自使用不同任务提示和观察范围，无需为每个角色单独部署一套权重。

LangGraph 承担状态图调度，项目自己实现控制策略、观察构建、Workspace 更新和消息授权。当前基础闭环为：

```text
图像 + 问题
    -> Workspace 摘要
    -> Controller/Policy 选择下一步
    -> Router 构建接收专家可见的上下文
    -> Expert 执行（可含视觉工具与私有演化）
    -> WorkspaceDelta 经 Reducer 提交
    -> 更新预算与日志，返回 Controller
    -> 有效 Finish 输出答案
```

主要实现：`control/`、`experts/`、`runtime/multi_agent.py`。

## 4. Workspace、空间描述和通信如何配合

Workspace 被保留为单次“图像 + 问题”任务的共享记录集合。它保存已提交的证据、答案候选、空间节点及制品引用，更新统一经过版本校验和 Reducer。Agent 私有候选、图像实体和大向量分别有自己的存储边界。

| 组件 | 回答的问题 | 实际职责 |
|---|---|---|
| Workspace | 系统已经确认并保存了什么？ | 保存可追溯记录与引用；统一更新版本 |
| 层次描述 | 证据属于图像哪里、处于什么尺度？ | Scene / Region / Group / Object、bbox、父子关系、来源 |
| ArtifactStore | 图片、crop 和向量实体在哪里？ | 存储物理制品，记录哈希、类型、shape 与来源 |
| Router / Inbox | 这次允许哪个专家看到什么？ | 按消息范围和模态授权披露记录 |
| ObservationBuilder | 专家本次实际拿到哪些输入？ | 从授权视图构建观察，限制跨专家信息暴露 |

单张影像不要求先构建完整 Spatial World Model。层级只按任务需要建立，中间层可以缺省；节点用于索引证据和 ROI，而不是承诺已经理解整幅图中的所有对象和关系。

通信采用受控共享与显式消息相结合的方式。共享存储减少重复保存；消息说明接收者、范围、模态和引用；读取端据授权加载内容。保存到 Workspace 不意味着所有专家立即可见。当前通信账本中的 text 是消息 envelope 的正则分词计数，structure 是 envelope 的 UTF-8 序列化字节，latent 按显式授权制品的描述大小计费。它们是实现口径下的代理量，不等同于全量展开上下文、模型 tokenizer 或网络抓包测得的流量；正式效率评测需要补全这些口径。

## 5. image latent 已经做到什么

已实现从本地视觉塔计算图像或 ROI 表示、池化为 5120 维 float32 向量、保存制品及哈希、绑定空间节点、授权读取和记录消费字节。单个此类向量为 20,480 字节。

接收专家目前在宿主程序中读取向量并计算诊断量；Reasoning 的 VLM 提示会记录已消费制品数量，但没有通过 projector 或 cross-attention 接收这些向量作为视觉 token。其看图能力仍来自图像、ROI 或联系图输入。因而现有证据支持 latent 管线已经接通，尚不足以证明连续视觉通信带来额外准确率提升。

2026-09-01 的真实多 Agent 单样本验证中，Text+Structure 与 latent 两个配置都答对风机数量 6。latent 配置保存七个向量，相关读取/通信计量为 143,360 字节。它验证实体消费和路径运行，样本量不足以评价增益。

还有一个复现问题：服务配置与别名写作 `evors/qwen3.8-27b-q4`，而视觉塔实现使用 `Qwen3_5VisionModel`，processor 指向 `Qwen/Qwen3.5-27B`。现阶段报告统一称“本地 Qwen 27B 级 Q4 服务”，正式论文实验前需核对模型、视觉塔、processor 和权重哈希的一致性，不能凭别名宣称同构表示空间已经验证。

## 6. 强化学习具体训练了什么

当前训练对象是一个 21 维状态输入、7 个动作输出的线性 softmax 路由策略，专家和 VLM 保持冻结。七个动作覆盖 Visual、Knowledge、两种 Reasoning 上下文（Text+Structure / Hybrid）、Verification、Finish 和 Abort。

有效动作掩码根据证据、候选、可用 latent 与已执行路径过滤动作。当前约束保留 Visual → 可选 Knowledge → Reasoning → Verification → Finish 的主要依赖关系，因此学习发生在受约束的流程内，不能解释成已学会任意专家重排。策略先选动作，环境随后执行对应专家并提交状态；事件记录把每个动作与前态和后续转移绑定，可验证改变策略是否真正改变执行轨迹。

训练使用 episode 级策略梯度、按任务维护的移动基线和熵正则。奖励同时包含答案正确性、证据与结构质量，以及专家、工具、模型、通信、重复和无效执行成本。质量项采用正奖励、成本采用负项，避免把“高质量证据”误作惩罚。

目前学习范围是专家激活、关键通信模态和终止，节点级 payload 由确定性构造逻辑产生，工具选择保留在专家内部。项目包含 verl 适配接口；保存的本地训练由独立 CPU 路由训练器完成，不能作为 PPO/GRPO/MAGRPO 或四卡 VLM 联合训练结果。

## 7. 保存下来的实验结果

### 7.1 真实 VLM 开发对照

数据为 20 个 VRSBench 开发样本，包含 10 个计数、10 个存在性问题。Phase 4 对照复用已冻结的 Direct 和感知输出，新执行下游 Reasoning / Verification，以控制视觉观察差异。

| 配置 | 正确数 | 准确率（Wilson 95% CI） | 平均模型调用 | 平均工具调用 | 记录的平均累计时间 |
|---|---:|---:|---:|---:|---:|
| Direct VLM | 10/20 | 50.0% [29.9%, 70.1%] | 1.00 | 0.00 | 8.05 s |
| Single Agent | 12/20 | 60.0% [38.7%, 78.1%] | 2.35 | 0.75 | 43.63 s |
| Multi Agent | 14/20 | 70.0% [48.1%, 85.5%] | 4.35 | 0.75 | 65.95 s |

这里的时间来自保存的感知阶段和增量执行记录，不能当作同一次完整重跑的部署吞吐。Multi 的平均通信为 260.8 文本 token、771.35 结构字节、0 latent 字节。因此，这组三臂结果也不能证明 latent 的贡献。

单 Agent 相对 Direct 增加 2 个正确答案，多 Agent 相对 Direct 增加 4 个，相对 Single 增加 2 个。对应精确 McNemar 双侧 p 值分别为 0.500、0.125、0.500，均未达到 0.05。它们证明开发目标在此样本集上实现，同时显示需要更大且独立的评测集。

成本也必须一起报告：Single 和 Multi 的平均模型调用分别为 Direct 的 2.35 倍和 4.35 倍。这组结果支持“更多观察和协作可能纠正部分错误”，不能写成“已经全面降低推理成本”。

证据：[三臂结果](../../evidence/2026-09-02_phase4_comparison/REPORT.md)、[原始汇总](../../evidence/2026-09-02_phase4_comparison/summary.json)、[单 Agent 门禁](../../evidence/2026-09-01_single_agent_perception_gate/README.md)。

### 7.2 路由学习机制实验

训练使用 18 个确定性任务、40 轮、seed 17，共 720 条轨迹；另有 6 个留出任务。初始与训练后策略使用同一留出集合。

| 指标 | 初始策略 | 训练后策略 |
|---|---:|---:|
| 平均奖励 | 0.235730 | 0.569049 |
| 留出任务正确率 | 66.7% | 100.0% |
| 平均 Agent 调用数 | 4.000 | 3.333 |

训练后计数和存在性任务跳过 Knowledge，分类任务保留 Knowledge；存在性任务选 Hybrid，计数任务选 Text+Structure。参数发生变化，专家指纹保持不变，评估不更新参数。这证明策略具有可执行影响；100% 是六个确定性任务的机制结果，不能解释为真实遥感数据集准确率。

两次真实 VLM 单样本 smoke 则验证 checkpoint 可以进入真实推理：Text+Structure 路径未消费 latent，Hybrid 路径授权并计入 20,480 字节。这些 smoke 与训练 fixture 的作用分别是验证接入和验证学习机制。

证据：[训练报告](../../evidence/2026-09-03_phase5_routing/REPORT.md)、[配置](../../evidence/2026-09-03_phase5_routing/run_config.json)、[Text+Structure smoke](../../evidence/2026-09-03_phase5_real_qwen_smoke/summary.json)、[Hybrid smoke](../../evidence/2026-09-03_phase5_real_qwen_hybrid_smoke/summary.json)。

## 8. 部署与协同开发交付

代码采用 Docker 环境和可挂载代码路径，为前置 ARM 机器与后续 Ascend 平台共用提供配置。2026-09-19 保存的发布证据显示实验镜像已上传 SWR，并注册到 ModelArts：

- 区域：`cn-southwest-2`。
- 镜像：`swr.cn-southwest-2.myhuaweicloud.com/evors-comm/training:phase5-20260919-d52590e8c17b`。
- 镜像摘要：`sha256:b1d866ae780b2685e15baf641a5f7bb25cfaf32dc674aebcb2581b17726d9779`。
- ModelArts 镜像 ID：`312ebef9-5fc6-4225-a0aa-a48ead950d39`；当时核验状态为 `ACTIVE`。

默认 `verify` 检查 NPU 与依赖；`routing` 运行当前 CPU 路由训练。输出可通过 `EVORS_OUTPUT_DIR` 指向已挂载的 `lws2`。镜像注册成功不代表已启动或完成四卡训练，也不代表当前挂载状态已重新核验。后续需保存平台作业日志、挂载、模型路径和设备利用率等实机证据。

发布时保存的测试记录为 321 passed、3 skipped，另排除三个会加载本地模型的重型测试；这是 2026-09-19 的历史发布验证。2026-09-25 的上传过程没有完成新一轮全量测试，本次报告也不声称进行了重跑。

部署说明：[ModelArts 使用文档](../MODELARTS_IMAGE.md)。发布证据：[镜像报告](../../evidence/2026-09-19_modelarts_image/REPORT.md)。

## 9. 下一阶段最值得做的工作

1. 核验并冻结模型身份、视觉塔、量化、processor、权重哈希和解码参数，解决配置命名不一致。
2. 固定独立训练、开发、测试划分；在同模型、同预算条件下重跑 Direct、Single、固定专家、规则路由和学习路由。
3. 分别消融多视图感知、候选演化、层次表示、受控披露和 latent，确定每一部分的独立贡献。
4. 若坚持将连续视觉通信作为核心贡献，实现接收端真正使用视觉 token 的接口，并以增益、成本和去除 latent 的反事实实验验证。
5. 在真实任务上训练并评价路由，报告多 seed、配对检验、置信区间和准确率-成本曲线。
6. 使用已发布的云端环境跑设备门禁，按实际需要启动大模型或多卡训练，并归档完整作业证据。

## 10. 本次配套文稿

论文初稿依据以上边界撰写，包含问题定义、方法、控制策略、真实开发结果、机制实验和后续评测方案。可编辑源文件在 [论文源码](../paper/evors_comm_draft_2026-10-08.tex)，成稿在 [PDF 论文初稿](../../output/pdf/EvoRS-Comm_Paper_Draft_2026-10-08.pdf)。已有历史论文文件保留，可用于比较设计演进。
