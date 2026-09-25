# VRSBench-20 真实数据 Smoke 实验

- **日期**: 2026-08-22
- **环境**: conda `rs-hmcomm`（Python 3.11.15），本机 llama.cpp `llama-server`，模型 `evors/qwen3.8-27b-q4`（Qwen3.8-27B Q4_K_M + mmproj Q8_0），endpoint `http://127.0.0.1:18080/v1`
- **GPU**: RTX 4070 Laptop 8GB
- **性质**: 管道 smoke + 初步对比。**scientific_claim: false**——20 条样本不足以支撑任何科学结论。

## 数据

- VRSBench val split（NeurIPS D&B 2024, CC-BY-4.0）：9,350 张图 / 标注
- 从 val 中分层随机抽样 20 题（seed=20260822），20 张不同图片：
  - `object_quantity` ×10：答案为纯数字 1–9，且问题文本不含该数字（防文本捷径）
  - `object_existence` ×10：Yes/No 各 5（平衡）
- 图片：512×512 PNG，从官方 `Images_val.zip` 解压实验所需 20 张
- manifest: [vrsbench20_manifest.jsonl](vrsbench20_manifest.jsonl)

## 实验设计

同一模型、同样 20 题，两个 arm：

| Arm | 方式 | 协议 |
|-----|------|------|
| `direct` | 整图 + 问题 → 单次 VLM 调用 | 直接回答 |
| `agent` | Phase 1 `SingleAgentRuntime` + crop 工具（require_tool=True，max_steps=4） | 多步推理 |

评分：quantity=预测中第一个数字精确匹配；existence=yes/no 归一化匹配。

## 结果

### 总体

| Arm | Accuracy | mean input tokens | mean output tokens | mean wall time |
|-----|----------|-------------------|--------------------|----------------|
| direct | **0.75** (15/20) | 325 | 2.0 | 10.3 s |
| agent | 0.45 (9/20) | 1529 (4.7×) | 89.8 | 74.2 s (7.2×) |

agent 臂平均每题调用 crop 工具 1.00 次。

### 分任务

| 任务 | direct | agent |
|------|--------|-------|
| object_quantity | 0.60 | 0.20 |
| object_existence | 0.90 | 0.70 |

## 主要观察

1. **管道端到端跑通**：真实 VRSBench 图 → 本地 Qwen3.8 → direct/agent 两臂 → 评分 → 落盘，40/40 记录无错误。
2. **在此 smoke 配置下，agent 臂（带 crop 工具）反而更差且贵 5–7 倍**。可能原因（未验证）：
   - crop bbox 由模型自行猜测，遥感小目标场景下经常裁错区域，引入误导性局部视图；
   - `require_tool=True` 强制走多步协议，输出格式漂移（如 val-007/008 答了空串）；
   - Q4 量化 + 8GB 显存限制下的推理能力边界。
3. **existence（0.90/0.70）显著好于 quantity（0.60/0.20）**——计数是硬任务，与 VRSBench 论文报告的 VLM 计数弱项一致。
4. 该结果**不构成对 agent 框架的否定**：本次 agent 未做任何针对遥感的 prompt/工具调优，crop 无先验引导；这正是后续演化（Phase 2）和协作路由（Phase 4）要改进的对象。

## 文件清单

- `events.jsonl` — 40 条逐样本记录（含答案/token/耗时/correctness）
- `summary.json` — 汇总指标
- `run_stdout.log` — 运行时标准输出
- `run_vrsbench20_smoke.py` — 实验脚本快照
- `vrsbench20_manifest.jsonl` — 抽样 manifest 快照
- `SHA256SUMS.txt` — 以上文件哈希

## 复现

```bash
# 1. 启动本地模型服务
powershell -File scripts/start_qwen38_local.ps1 -Action Start
# 2. 跑实验
python scripts/run_vrsbench20_smoke.py --output-dir <EV_DIR>
```

依赖数据：`data/vrsbench/val/images/*.png`（来自 VRSBench `Images_val.zip`）、标注 zip 同目录。

## 已知问题

- 首次运行 summary 写入时因 backend 属性名笔误（`base_url`→`endpoint`）中断；40 条数据已完整落盘，summary 由 events.jsonl 重建，脚本已修复。两臂数据均产生于同一次连续运行。
