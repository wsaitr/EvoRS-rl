# 单 Agent 感知增强门禁

> 状态：已通过开发门禁
>
> 日期：2026-09-01

## Context

EvoRS-Comm 的 Agent 必须通过主动视觉观察增强同一 VLM 的遥感感知，
而不是仅增加文本协议、路由或重复推理。当前 Direct VLM 与旧 Generalist
已经能运行，但旧 Generalist 的盲裁剪和严格 JSON 循环会降低准确率。

## Goal

在固定的 VRSBench-20 开发门禁上，使用同一个 Qwen3.8-27B-Q4 服务，
让单 Agent 的正确题数严格高于单次整图 Direct VLM。

## Non-Goals

- 本门禁不实现多 Agent、Controller、RL 或 verl；
- 不使用参考答案、GT bbox 或数据集标注控制运行时行为；
- 不更换更强模型来获得表面提升；
- 20 样本只作为开发门禁，不作为论文统计显著性结论。

## Requirements

- Direct 与 Agent 使用相同模型别名、模型修订、服务参数和评分函数；
- Direct 只进行一次整图问答；
- Agent 只能根据图像、问题、任务类型和自身中间结果选择观察动作；
- Agent 必须向 VLM 提供真实新增视觉信息，例如确定性分块、放大 ROI、
  Contact Sheet 或带框整图，不能用 bbox 文本或 latent 标量冒充视觉增强；
- 所有模型调用、输入输出 token、工具调用和耗时必须计入各自 arm；
- 逐样本记录最终答案、正确性、视觉动作、模型调用和错误；
- 运行时不得读取 `reference_answer`，参考答案只进入离线评分函数。

## Acceptance Criteria

- [x] Direct 与 Agent 均覆盖同一20个样本且运行错误为0；
- [x] `agent_correct > direct_correct`，至少净增1题；
- [x] 计数题和存在性题分别报告正确率；
- [x] 结果包含模型身份、manifest/config hash 与完整 JSONL；
- [x] 新增视觉工具和选择逻辑具有离线单元测试；
- [x] 完整项目测试通过。

## Gate Result

2026-09-01 使用 `evors/qwen3.8-27b-q4`、固定 seed `20260831`
完成配对运行：

| Arm | Overall | Quantity | Existence | Errors | Model calls |
|---|---:|---:|---:|---:|---:|
| Direct | 10/20 (50%) | 4/10 | 6/10 | 0 | 20 |
| Single Agent | 12/20 (60%) | 6/10 | 6/10 | 0 | 47 |

开发门禁结论：Single Agent 净增 2 题，绝对准确率提升 10 个百分点，
且未将 Direct 已答对的样本改错。完整项目测试为 `319 passed, 3 skipped`。

证据目录：`evidence/2026-09-01_single_agent_perception_gate/`。
该结果仍然只是小样本开发门禁，不升级为论文级统计结论。

## Implementation Plan

1. 固定 Direct 提示、采样参数与现有评分器；
2. 实现一个单 Agent 主动感知循环，优先从无重叠多尺度分块和
   Contact Sheet 开始；
3. 对计数题使用放大分块/候选复核，对关系与存在性题保留整图上下文；
4. 在同一次实验中交错运行 Direct 与 Agent，记录真实成本；
5. 未超过 Direct 时只调整感知动作和视觉融合，不扩展多 Agent/RL；
6. 达标后固化 evidence，并在后续更大未见集上复验。

## Verification

```powershell
$env:PYTHONPATH = "src"
python scripts/run_single_agent_perception_gate.py `
  --manifest configs/experiments/vrsbench20_manifest.jsonl `
  --output-dir evidence/<run-id>
python -m pytest -q
```
