# EvoRS-Comm 论文目标对齐文档

日期: 2026-08-27
用途: 每次动手前先核对这份文档，防止走偏

---

## 一、论文核心主张（一句话）

**通过"层次空间索引 + 多模态通信"增强 VLM 的感知能力，使其在遥感 VQA 任务上显著优于直接 VLM 问答。**

关键词：
- **增强感知**（不是取代 VLM，是让 VLM 看到更多/更清）
- **层次空间索引**（Scene → Object，不是扁平列表）
- **多模态通信**（text + structure + **视觉** latent，不是纯文本）

---

## 二、反例检验（什么不是我们的目标）

❌ **不是**：多个 VLM 实例互相聊天
❌ **不是**：把 bbox 坐标写成文本塞进 prompt
❌ **不是**：把 latent 向量算 L2/余弦后写数字进 prompt
❌ **不是**：verification 看 reasoning 的答案然后说"接受"
❌ **不是**：knowledge 专家说一句"请考虑地理知识"

这些都被证明无效（消融实验），**禁止**作为 agent 的核心机制。

---

## 三、论文创新点 vs 实现机制 vs 实验验证（逐一对齐）

### 创新点 1：层次空间表示 → 增强 VLM 感知

**目标**：把 VLM 看不清楚的小目标，通过空间索引 + 接触图(contact sheet) 的方式**放大给 VLM 看**。

**正确实现**：
1. Visual expert：整图粗检测 → 得到 N 个 (label, bbox)
2. 工具层：对每个 bbox 确定性裁剪（无 VLM，100% 可靠）
3. 组装 contact sheet：把 N 个 ROI 拼成一张编号图
4. Reasoning expert：**看着 contact sheet 回答**（真视觉输入）

**验证指标**：
- 小目标计数准确率 > direct（VLM 现在能"看清"了）
- 对比实验：`hierarchical_contact_sheet` vs `direct` vs `flat_collab`

### 创新点 2：多模态通信（text + structure + 视觉 latent）

**目标**：不同专家之间传递的**不只是文本**，还有结构化空间和视觉 latent。

**关键澄清**：
- text：任务描述、答案（必要但不够）
- structure：bbox、节点关系（让下游知道"哪里有目标"）
- **视觉 latent**：应该是**图像本身**（ROI 图），不是标量数字

**禁止**：把 5120 维向量算 norm/cosine 写成 `0.978` 进 prompt —— VLM 不读数字。
**允许**：把 latent 作为 visual tokens 通过 projector 注入（需要训练，论文未来工作）。

**验证指标**：
- 消息中实际携带的模态比例（text/structure/visual 各占多少字节）
- 有 visual latent vs 纯 text 的准确率对比（H2）

### 创新点 3：受控信息披露

**目标**：不是把 workspace 全量塞给下游，而是按任务/角色精准披露。

**实现**：`disclosure_mode` 参数（full vs scoped），scoped 模式只给推理需要的 ref 集合。

**验证指标**（H6）：
- scoped vs full 的准确率-成本帕累托曲线
- 不能只报"通信成本降了"，要证明"准确率没降或降得少"

### 创新点 4：学习式路由

**目标**：根据历史样本学习路由策略，优于固定规则。

**当前问题**：训练记忆全部用规则序列，无差异 → 退化为规则。

**需要**：让 learned_route 在真实差异路由的数据上训练（如：计数题跳过 knowledge，分类题走 knowledge+reasoning）。

**验证指标**（H3/H5）：learned vs rule 的准确率对比（控制相同 expert 集合）。

### 创新点 5：单 agent 内部演化

**目标**：推理前先生成多个候选答案，用演化选择最优（1+λ ES）。

**当前问题**：未集成到协作管线。

**验证指标**（H4）：with-evolution vs without-evolution（同管线，仅开/关演化）。

---

## 四、P0 修复清单（按创新点对齐）

### 对应创新点 1（感知增强）—— 最高优先

- [ ] **R1: 实现 ContactSheetTool**
  - 输入：图 + N 个 (label, bbox)
  - 输出：contact sheet 图像（确定性、几何、无 VLM）
  - 单元测试：拼图正确性、编号正确性
- [ ] **R2: Reasoning expert 改为接收 contact sheet 作为视觉输入**
  - 替代当前 bbox 文本列表
  - 同时保留实例表（`#1 roundabout`, `#2 roundabout`）作为结构化文本
- [ ] **R3: 实验验证 R1+R2**
  - `hierarchical_contact_sheet` arm vs `direct` vs `flat_collab`
  - 关键指标：小目标计数准确率（此前 0.40 的计数瓶颈能否打破）

### 对应创新点 2（多模态通信）

- [ ] **R4: 删除 reasoning prompt 中的 latent 标量注入**
  - 当前 `latent_lines` 写 `0.978` 进 prompt，已证明无效
  - latent 暂时只作为 workspace 制品存储，待后续有 projector 时再用
- [ ] **R5: 把 ROI 图作为 visual artifact 传递**
  - contact sheet 本身也是 artifact
  - reasoning 通过 scoped view 收到 ROI 图的 visual ref，作为真实视觉输入

### 对应创新点 3（受控披露）

- [ ] **R6: H6 帕累托曲线**
  - 多 disclosure_mode（full / scoped-strict / scoped-loose）× 多 routing 策略
  - 报告 accuracy-vs-comm-cost 曲线，不只看单点
- [ ] **R7: 修复 flat/hierarchy 未隔离问题**
  - flat arm 不应跑 spatial detection，否则不是 flat
  - 重命名 arm 使语义清晰

### 对应创新点 4（学习路由）

- [ ] **R8: 训练记忆差异化**
  - 当前全是 `visual→reasoning→verification`
  - 需要：计数题跳过 knowledge；分类题走 knowledge+reasoning；存在性题只走 visual+reasoning
  - 重新跑 learned_route arm

### 对应创新点 5（演化）

- [ ] **R9: 演化集成**
  - reasoning 专家内部：生成 N 个候选答案 → 适应度评估 → 选最优
  - 适应度：VLM-as-judge（独立于候选答案的新 VLM 调用）
  - 验证：with-evolution vs without-evolution 同管线对比

---

## 五、实验口径修正（独立于创新点）

这些不是创新点，但影响实验有效性：

- [ ] **E1: 所有专家调用计入 token 成本**
  - 当前只有 reasoning 写 `model_cost` evidence，视觉/验证/知识专家的 token 没统计
- [ ] **E2: 专家异常写入事件流，评测把异常计错**
  - 当前裸 except → NO_PROGRESS，错误样本与真失败混在一起
- [ ] **E3: verification 改为独立作答**
  - 当前 verifier 看 reasoning 的答案后 accept/reject → anchoring 偏置
  - 改为：verifier 独立看整图+contact sheet 给出答案 → 程序比对

---

## 六、不做清单（防止范围蔓延）

| 事项 | 理由 |
|------|------|
| 训练专用 detector (YOLOv8 等) | 工程量大，偏离论文创新点；本文聚焦"通信增强 VLM 感知"，不是"替换 VLM" |
| 训练 projector 把 latent 转成 visual tokens | 需要 GPU 集群和训练数据，留作未来工作 |
| 扩到 1000+ 样本 | 当前 20 样本足够做机制验证；100+ 用于显著性，1000+ 留给最终论文 |
| RL 训练路由策略 (H5 完整) | 当前样本量和算力不支持，留作未来工作 |
| 多模型异构 agent | 当前单 Qwen3.8-27B-Q4，论文先验证机制，异构留作扩展 |

---

## 七、每次动手前的核对清单

1. 这个改动对应上面哪个创新点？
2. 改动后 VLM 的感知是**增强**了还是**没变**还是**变差**？
   - 增强 ✅：VLM 现在能"看到"之前看不到的信息（如 ROI 图）
   - 没变 ⚠️：VLM 看到的和 direct 一样（如 clean prompt）
   - 变差 ❌：VLM 被无关信息干扰（如 bbox 文本、latent 数字）
3. 改动是否能让对应假设的实验**干净地**测出来？

---

## 八、对齐检查点

当前状态 vs 目标的差距：

| 创新点 | 当前状态 | 目标状态 | 差距 |
|--------|---------|---------|------|
| 1. 层次感知 | 文本前缀（失败） | contact sheet（待实现） | **R1+R2+R3** |
| 2. 多模态通信 | 标量数字（失败） | 视觉 artifact + 结构化索引 | **R4+R5** |
| 3. 受控披露 | 有数据但 arm 未隔离 | 帕累托曲线 | **R6+R7** |
| 4. 学习路由 | 退化为规则 | 差异路由记忆 | **R8** |
| 5. 演化 | 未集成 | with/without 对比 | **R9** |
| 口径 | 成本漏算 | 全量统计 | **E1+E2+E3** |

**P0 = R1+R2+R3+E1+E2** （先让创新点 1 立起来 + 口径修正）
