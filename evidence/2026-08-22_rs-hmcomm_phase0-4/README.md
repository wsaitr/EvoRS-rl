# EvoRS-Comm Phase 0–4 统一环境验证证据

- **验证日期**: 2026-08-22
- **执行环境**: conda env `rs-hmcomm`（`D:\ZZZAI\Anaconda\Anaconda\envs\rss-hmcomm`，Python 3.11.15），项目以 editable 模式安装
- **项目根目录**: `f:\project\yaogan_lunwen`
- **分支**: `codex/phases0-4`
- **背景**: 此前 codex 在系统 Python 3.12（`D:\ZZZIDE\python\Python312`）中以 pytest 临时目录方式验证过代码；本次为首次在统一约定的 `rs-hmcomm` 环境中运行实验脚本并留存产物。

## 结果总览

| Phase | 脚本 | status | 证据文件 |
|-------|------|--------|----------|
| 0 | `scripts/validate_phase0.py` | `contract_smoke_passed` | [phase0/events.jsonl](phase0/events.jsonl) |
| 1 | `scripts/run_phase1_generalist.py`（mock 后端） | `passed`, gate_passed=true | [phase1/events.jsonl.summary.json](phase1/events.jsonl.summary.json) |
| 2 | `scripts/run_phase2_evolution.py` | `passed`, gate_passed=true | [phase2/events.jsonl.summary.json](phase2/events.jsonl.summary.json) |
| 3 | `scripts/run_phase3_representation.py`（latent 启用） | `passed` | [phase3/representation-summary.json](phase3/representation-summary.json) |
| 4 | `scripts/run_phase4_collaboration.py`（fixed/all_agent/rule × 2 任务） | `passed`, failures=[] | [phase4/collaboration-summary.json](phase4/collaboration-summary.json) |

单元测试：**262 passed, 1 skipped**（排除 `tests/live/` 真机测试），完整日志见 [pytest_full.log](pytest_full.log)。

## 文件清单

- `environment.txt` — rs-hmcomm 环境 Python 及关键依赖版本（langgraph 1.2.11 等）
- `phase{0..4}/run_stdout.txt` — 各脚本标准输出（含完整 JSON 摘要）
- `phase{0..4}/…` — 实验产物（事件日志 / summary）
- `pytest_full.log` — 全量单元测试日志
- `SHA256SUMS.txt` — 以上全部文件的 SHA256 校验清单
- `README.md` — 本说明

## 复现命令

在项目根目录、激活 `rs-hmcomm` 后依次执行：

```bash
python scripts/validate_phase0.py --output <EV>/phase0/events.jsonl --overwrite
python scripts/run_phase1_generalist.py --output <EV>/phase1/events.jsonl --overwrite
python scripts/run_phase2_evolution.py   --output <EV>/phase2/events.jsonl --overwrite
python scripts/run_phase3_representation.py --output <EV>/phase3/representation-summary.json --overwrite
python scripts/run_phase4_collaboration.py  --output <EV>/phase4/collaboration-summary.json --overwrite
python -m pytest tests/ --ignore=tests/live -v
```

## 边界说明（防止过度声明）

- Phase 1 本次为 **mock 后端**验证（控制流）；真实 VLM（Qwen3.8-27B）的 live gate 通过证据见 `outputs/phase1/qwen38-live-final-current-events.jsonl.summary.json`（2026-08-21）。
- 所有 summary 中 `scientific_claim: false`——本批验证仅证明**机制可运行**，不构成任何科学结论。
- 未覆盖项：Phase 2 matched-budget 对比（推迟至 Phase 6）、ARM 真实 encoder smoke、ARM Docker 完整 smoke。
