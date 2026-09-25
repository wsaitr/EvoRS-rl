# EvoRS-Comm 真实管线实验：H1/H6 首轮可测数据

- **日期**: 2026-08-23
- **环境**: conda `rs-hmcomm`，本机 llama-server（Qwen3.8-27B Q4_K_M + mmproj Q8_0，`--embeddings --pooling last`），RTX 4070 Laptop 8GB
- **数据**: VRSBench val 20 样本（同 2026-08-22 manifest，seed=20260822）
- **性质**: 机制真实性验证 + 首轮性能数据。**scientific_claim: false**（n=20，单种子）

## 本轮完成的机制升级（占位符 → 真实）

| 模块 | 之前 | 现在 |
|------|------|------|
| 视觉专家 | 硬编码全图 bbox 的规则 fixture | `VLMVisualExpert`：Qwen 真实检测，输出 Scene→Object 层次树 + 归一化 bbox（画布推断归一化） |
| 推理专家 | 固定字符串 `requires_model_reasoning` | `VLMReasoningExpert`：基于空间树上下文真实答题 |
| 验证专家 | 无条件复制候选 | `VLMVerificationExpert`：VLM 复核，可修正答案 |
| 知识专家 | 内置关键词规则 | `VLMKnowledgeExpert`：文本 VLM 生成领域指引 |
| Latent 编码器 | 12 维像素统计 | `QwenMmprojLatentEncoder`：mmproj 视觉塔 5120 维 last-token 特征（L2 归一化），支持 ROI |
| 路由策略 | 仅 fixed/all/rule | 新增 `LearnedRoutingPolicy`：记忆+相似度加权投票，规则回退 |

关键工程点：
- 图像通过 `ImageStore`（进程内 ref→PIL 映射）传给专家，符合契约"二进制不进 Workspace"
- VLM bbox 坐标系（内部 smart-resize 画布）通过响应内最大坐标推断画布（28 的倍数对齐）归一化
- 专家 VLM 调用成本通过 `model_cost` 类型 Evidence 记入 Workspace，实验层聚合
- llama-server 增加 `--embeddings --pooling last`（启动脚本已更新），原生 `/embedding` 端点支持 `image_data`

## 实验设计（三臂，同一后端同一批样本）

| Arm | 机制 | 通信内容 |
|-----|------|----------|
| `direct` | 整图单次问答 | 无 |
| `flat_collab` | 4 VLM 专家协作，全披露 | text+structure |
| `hierarchical_comm` | 同上 + 空间树 ROI latent | text+structure+**latent** |

## 结果

### 总体（n=20）

| Arm | Acc | VLM tok_in/题 | msg text | msg struct | latent B/题 | 耗时/题 |
|-----|-----|---------------|----------|------------|-------------|---------|
| direct | 0.50 | 317 | 0 | 0 | 0 | 8.6s |
| flat_collab | 0.55 | 387 | 217 | 610 | 0 | 49.1s |
| hierarchical_comm | **0.65** | 377 | 262 | 703 | 98,304 | 54.7s |

### 分任务

| 任务 | direct | flat_collab | hierarchical_comm |
|------|--------|-------------|-------------------|
| object_quantity | 0.30 | 0.30 | **0.50** |
| object_existence | 0.70 | 0.80 | 0.80 |

### 逐题亮点（支持 H1 方向的样本）

- val-002（ref=1）: direct✗ flat✗ **hier✓** —— ROI latent 帮助聚焦
- val-006（ref=2）: direct✗ flat✗ **hier✓**
- val-000（ref=2）: direct✗ **flat✓ hier✓**
- val-015/016: direct✗（答"Yes."）flat✓ hier✓ —— 协作验证修正了直接问答的格式/判断错误

## 初步观察（不构成结论）

1. **方向与 H1 一致**：hierarchical > flat > direct（0.65 > 0.55 > 0.50），计数任务上层次化+latent 优势最大（0.50 vs 0.30）
2. **成本合理**：hierarchical 相比 flat 只增加 ~5s 和 ~20KB latent 带宽，换来 +10pp 准确率
3. **计数仍是硬任务**：所有臂在 quantity 上 ≤0.50；val-008（车辆计数）全部大错（10-15 vs ref=1），暴露检测召回问题
4. **H3 未测**：本轮 `learned_route` 臂未运行（需要先积累记忆 episode，设计上属于第二轮）；LearnedRoutingPolicy 机制已就位
5. n=20 单种子下 10pp 差异无统计显著性；扩样本+多种子后才能进论文

## 文件清单

- `events.jsonl` — 60 条逐样本记录（三臂×20）
- `summary.json` — 汇总指标
- `run_stdout.log` — 运行日志
- `run_real_pipeline.py` — 实验脚本快照
- `vlm_mechanisms_snapshot.py` / `qwen_latent_snapshot.py` / `learned_policy_snapshot.py` — 机制代码快照
- `vrsbench20_manifest.jsonl` — 数据 manifest
- `SHA256SUMS.txt` — 全部文件哈希

## 复现

```bash
# 服务器需带 --embeddings --pooling last（start_qwen38_local.ps1 已更新）
powershell -File scripts/start_qwen38_local.ps1 -Action Start
python scripts/run_real_pipeline.py --manifest configs/experiments/vrsbench20_manifest.jsonl --output-dir <EV_DIR>
```

## 下一步

1. 扩样本至 100+ 并跑 3 种子，检验 0.65 vs 0.55 是否稳定
2. 加入 `learned_route` 臂：先在 train split 上积累记忆，val 上评测（H3/H5）
3. 计数任务专项：检测召回是瓶颈，考虑多 crop 投票或检测专用 prompt
4. H6 消融：hierarchical_comm vs 全披露（当前 flat 即近似全披露，可再细分披露粒度）
