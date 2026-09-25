# 上下文噪音消融实验（hierarchical_clean）

- **日期**: 2026-08-25
- **目的**: 验证"multi-agent 落后 direct 的主因是 reasoning prompt 中的噪音注入"
- **设计**: hierarchical_clean 臂 = 与 hierarchical_comm 相同管线，但 reasoning
  expert 不接收检测文本与 latent 标量诊断（include_detection_context=False）
- **主要结果**:
  - hierarchical_clean 0.70 ≥ direct 0.60（存在性任务 0.90 vs 0.70）
  - 带噪音的 hierarchical_comm 0.65；计数任务上噪音直接导致 val-006 翻转错误
- **结论**: 协作结构本身不劣于 direct；伤害来自错误检测文本与无意义的 latent 数字
- 详细分析见 `../2026-08-23_real_pipeline_native_latent/REPORT.md` §2.5
