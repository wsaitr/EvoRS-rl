# EvoRS Phase 4 配对对照实验规格

> 状态：实现完成，20 样本真实配对实验已通过
>
> 日期：2026-09-02

## Context

现有证据已经证明：Direct VLM 为 10/20，主动感知 Single Agent 为 12/20；
Phase 4 Multi Agent 在一个真实样本上完成了 Text/Structure 与 latent 双路径。
当前缺口是同一批样本上的 Direct、完整 Single Agent 和 Multi Agent 准确率—成本
对照，以及可恢复、可追溯的实验产物。

## Experimental Design

采用 `configs/experiments/vrsbench20_manifest.jsonl` 的全部 20 个固定样本。

三条 arm：

1. `direct_vlm`：复用已保存的真实 whole-image VLM 输出；
2. `single_agent`：复用同一组真实主动感知调用，重新执行 Workspace、层次表示和
   私有 evolution，得到完整 Single Agent 结果；
3. `multi_agent`：以同一个冻结的主动感知结果驱动 EvolvingVisualExpert，再真实
   调用 ReasoningExpert 和 VerificationExpert。

这是 paired frozen-perception comparison：Direct 和 Single 的原始 VLM 响应不重采样，
Multi 只新增协作层调用。这样视觉证据完全一致，差异来自表示、路由和协作，而不是
Visual 重采样噪声。

## Goals

- 计算三个 arm 的 overall、counting 和 existence accuracy；
- 统计 model/tool calls、token、wall time 和通信成本；
- 给出 Multi 相对 Direct/Single 的逐样本 win/loss/tie；
- 保存完整逐样本输入、输出、Workspace、transition、message 与成本；
- 支持中断后续跑，不覆盖已经完成的真实样本；
- 生成机器可读 JSON/JSONL/CSV 和人可读实验报告。

## Non-Goals

- 不把 20 样本结果表述为论文最终结论；
- 不在本轮训练 RL 或调参；
- 不重新采样 Direct/Visual 输出；
- 不把 reference answer 放入任何 Agent 的 Task Context 或 prompt；
- 不用 reference answer 决定路由、候选、验证或终止。

## Requirements

1. 固定 manifest、模型配置、seed 和 baseline evidence SHA-256；
2. baseline event 必须覆盖每个样本的 direct/agent 两条记录且无错误；
3. replay perception 必须逐字段保留 view、answer、tokens、latency 和 model ID；
4. Single 与 Multi 必须使用相同的 replayed perception result；
5. Single 必须真实重建 Workspace、Scene/Group/Object 和 private evolution；
6. Multi Task metadata 只能包含 dataset sample ID，不含答案或标注；
7. Multi 必须使用 EvolutionController + RuleRoutingPolicy + scoped Router；
8. 每完成一个样本立刻 append JSONL 并原子更新 checkpoint；
9. 汇总必须同时报告 accuracy 和平均/总成本，不能只报准确率；
10. paired comparison 必须列出 changed samples，不能只给聚合比例；
11. error/未完成样本不得静默计为错误答案，必须单独统计；
12. 结果报告必须声明 frozen-perception 和小样本限制。

## Acceptance Criteria

- [x] 20 个样本三条 arm 均有可解析记录；
- [x] 完整 Single Agent 的答案与保存的 Active-Perception 输出一致或有可解释 evolution 变化；
- [x] Multi 每个成功 episode 均有可回放 transition/message 日志；
- [x] reference answer 不出现在 Agent Task metadata；
- [x] 生成 `events.jsonl`、`checkpoint.json`、`summary.json`、`results.csv`、
      `accuracy_cost.csv` 和 `REPORT.md`；
- [x] 汇总包含 overall/counting/existence accuracy 与 paired win/loss/tie；
- [x] 汇总包含 model/tool calls、tokens、wall time和三种通信成本；
- [x] 实验脚本离线测试及项目全量测试通过。

## Verification Result

20 个固定 VRSBench validation 样本全部完成，三个 arm 均无错误：

| Arm | Accuracy | Wilson 95% CI | Mean model calls | Mean wall time |
|---|---:|---:|---:|---:|
| Direct VLM | 10/20 = 50% | [29.9%, 70.1%] | 1.00 | 8.05 s |
| Complete Single Agent | 12/20 = 60% | [38.7%, 78.1%] | 2.35 | 43.63 s |
| Multi Agent | 14/20 = 70% | [48.1%, 85.5%] | 4.35 | 65.95 s |

任务拆分：

- object quantity：Direct 4/10，Single 6/10，Multi 6/10；
- object existence：Direct 6/10，Single 6/10，Multi 8/10。

配对结果：

- Single vs Direct：2 wins / 0 losses / 18 ties，exact McNemar p=0.500；
- Multi vs Direct：4 wins / 0 losses / 16 ties，exact McNemar p=0.125；
- Multi vs Single：2 wins / 0 losses / 18 ties，exact McNemar p=0.500。

Multi 相对 Single 在 `vrsbench-val-015`、`vrsbench-val-016` 上把错误的
`Yes` 修订为正确的 `No`。`vrsbench-val-004` 从错误的 `3` 改为错误的 `4`，
属于 changed-but-still-wrong，不计 accuracy loss，但已保留在失败分析中。

平均通信成本为 260.8 text tokens、771.35 structure bytes、0 latent bytes；
本轮主对照固定为 Text + Structure，latent 机制证据见前一 Phase 4 双路径 smoke。

完整性审计：20 个唯一样本、60 个 arm 结果无错误；Single Workspace 均为
version 2，Multi 均为 version 3；Multi 均包含 3 个 Agent、2 条 routed message、
4 个 transition；Task metadata 无 reference answer，答案所有权正确，Workspace
无 image/crop/latent 二进制值。全量测试为 `329 passed, 3 skipped`。

证据目录：`evidence/2026-09-02_phase4_comparison/`。

结果证明当前 20 样本开发集上存在 50%→60%→70% 的单调改善，但置信区间重叠，
配对检验未达到常用显著性阈值，不能替代 Phase 6 的多 seed、大样本论文实验。

## Implementation Plan

1. 实现 baseline evidence 校验与 `RecordedPerceptionAgent`；
2. 实现完整 Single Agent 离线重建；
3. 实现 resumable Multi Agent paired runner；
4. 实现逐样本 append、checkpoint 和聚合报告；
5. 增加 fake-backend 小型端到端测试；
6. 执行 20 样本真实实验并核对证据文件；
7. 回填验收结果与限制。

## Verification

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_phase4_comparison.py
python scripts/run_phase4_comparison.py --help
python -m pytest -q
```
