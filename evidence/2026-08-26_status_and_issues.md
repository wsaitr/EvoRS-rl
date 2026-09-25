# EvoRS-Comm 阶段性成果与问题清单

日期: 2026-08-26（覆盖 2026-08-22 至 2026-08-26 全部工作）

---

## 一、已取得的成果

### 1. 工程基础（全部真实可用）

| 组件 | 状态 | 位置 |
|------|------|------|
| 契约层（Task/Evidence/Claim/Candidate/CostVector/Trace） | ✅ 完备，262 个契约测试 | `src/rs_hmcomm/contracts/` |
| 单 agent 运行时（预算、重放、事件溯源） | ✅ 可运行 | `runtime/single_agent.py` |
| 多 agent LangGraph 运行时（控制器路由、消息信封、scoped 披露） | ✅ 可运行 | `runtime/multi_agent.py` |
| 层次空间表示（Scene→Object，bbox 归一化，1024 画布标定） | ✅ 真实数据验证 | `spatial/representation.py` + `experts/vlm_mechanisms.py` |
| 真实 latent 编码器（Qwen3.5 视觉塔进程内推理，5120 维，内容敏感） | ✅ 三项内容敏感性测试通过 | `artifacts/qwen_native_latent.py` |
| 确定性工具（CropTool / GridCropTool） | ✅ 几何单测全过 | `tools/` |
| 测试套件 | ✅ 298 passed, 1 skipped | `tests/` |

### 2. 实验发现（有数据支撑）

**F1 — 噪音消融（最重要）**：协作管线在干净上下文下 **不弱于 direct**。
存在性任务上 hierarchical_clean 0.90 vs direct 0.70。伤害来自 reasoning
prompt 中注入的错误检测文本与无信息量的 latent 标量数字，而非协作结构本身。

**F2 — verification 专家有真实贡献**：val-011/015/016 三题 direct 错、协作对。

**F3 — H6 初步证据**：scoped disclosure 节省 11% 通信成本（msg_text -12%），
代价 8% 准确率下降——披露粒度与准确率的权衡关系首次量化。

**F4 — 层次索引↔latent 挂载链路打通**：每个 Object 节点经
`visual_artifact_refs` 指向对应 ROI latent，bbox 与 latent roi 一致，
evidence 携带 scene latent 引用——论文声称的"层次空间索引的多模态通信"
在机制层成立。

**F5 — 单点视觉确认不稳定**：同一 crop 的 zoom-confirm verdict 在多次调用间
Yes/No 波动——4-bit 量化 VLM 的细粒度判断不可靠，硬过滤方案不可行。

### 3. 设计审查结论（自查 + codex 交叉验证，详见 [agent_design_audit.md](../2026-08-26_agent_design_audit.md))

单 agent 落后 direct 的根因是三个可修复的设计错误：
1. 无检测能力支撑的盲 crop（裁剪让信息变差）
2. 整图+crop 双图输入混淆（"不要重复计数"规则反而误导）
3. JSON 协议负担挤占 4-bit 模型的视觉推理能力

---

## 二、当前问题清单

### A. 直接影响实验结论有效性的（必须修）

| # | 问题 | 影响 |
|---|------|------|
| A1 | reasoning 从不看 ROI 图，只看 bbox 文本——"层次协作"实为 direct+文本前缀 | H1 无法真正检验 |
| A2 | flat vs hierarchy 两臂没有干净隔离（flat 也跑同样的检测；hierarchical_comm 实际 full disclosure） | H1 对照失效 |
| A3 | 成本漏算：只有 reasoning 写 model_cost，visual/verification 的 token 不进统计 | 所有成本对比失真 |
| A4 | verifier 解析失败仍标 VERIFIED | "验证收益"被高估 |
| A5 | runtime 裸 except 吞专家异常 → NO_PROGRESS | 错误样本与真失败混在一起 |

### B. 已知但未修的机制缺陷

| # | 问题 | 说明 |
|---|------|------|
| B1 | fallback 全图 OBJECT 节点进入 reasoning 上下文 | 存在性偏置 + 计数锚定 |
| B2 | nested crop 的 source_bbox 记录局部坐标未合成全局坐标 | 多级裁剪时坐标错误 |
| B3 | knowledge 专家在主臂中从不执行；其输出即使执行也不进 reasoning prompt | 四专家实为三专家 |
| B4 | bbox 画布假设固定 1024，矩形图的 smart-resize 可能不准 | 检测框系统性偏移风险 |
| B5 | 演化(H4)未集成到协作管线 | 六个假设只测了两个 |

### C. 模型能力边界（不是工程问题）

- 计数任务四臂全部 ~0.40-0.50：Qwen3.8-Q4 的细粒度计数能力短板
- zoom-confirm verdict 波动：同 crop Yes/No 不稳定
- 20 样本下 ±0.1 的运行方差（VLM 输出格式漂移）

---

## 三、下一步修复路线（P0 → P2）

```
P0（恢复实验有效性）
├── R1: reasoning 保持 clean prompt（已有消融支撑），ROI 改为"带编号框的
│      整图重注入"而非文本列表 —— 对齐 GeoChat 式做法
├── R2: verification 独立作答后程序比对（去 anchoring）
└── R3: fallback 节点标记 non_visual，不进 reasoning 上下文

P1（口径修正）
├── R4: arm 重命名 + flat 臂去 spatial 化（真隔离层次变量）
├── R5: 所有专家调用写 model_cost evidence
└── R6: 专家异常记录到事件流，评测把异常计错并报告 coverage

P2（补齐假设覆盖）
├── R7: 2x2 消融分离 boxes / latent 各自贡献
├── R8: 演化集成进协作管线（H4）
└── R9: 扩样 100+ 条验证 F1/F3 显著性
```

## 四、证据索引

```
evidence/
├── 2026-08-22_rs-hmcomm_phase0-4/          # Phase 0-4 统一环境验证
├── 2026-08-22_vrsbench20_smoke/            # 首轮真实数据两臂 smoke
├── 2026-08-23_real_pipeline_h1h3h6/        # 三臂管线（旧 latent，含 bug 期数据）
├── 2026-08-23_real_pipeline_native_latent/ # 五臂主实验 + REPORT.md（含噪音消融 §2.5）
│   └── scoped_fixed/                       # scoped 臂修复重跑
├── 2026-08-23_noise_ablation/              # hierarchical_clean 消融
├── 2026-08-25_zoom_counting/               # zoom-confirm 探索臂
├── 2026-08-26_agent_design_audit.md        # 本轮设计审查结论 ★
└── (本文件)                                 # 成果与问题总览 ★
```
