# Zoom-confirm 臂实验（探索性）

- **日期**: 2026-08-25
- **设计**: visual expert 对每个候选 bbox 做确定性裁剪放大（min_side=112），
  VLM yes/no 确认后再入层次树
- **结果**: acc=0.65（quantity 4/10, existence 9/10）；tok_in=416, wall=76s
- **关键发现**: 同一 crop 的 confirm verdict 在多次调用间不稳定（Yes/No 波动），
  单点确认不可用作硬过滤；计数能力是量化模型本身的短板
- **处置**: 该臂未纳入主线对比；网格扫描计数模块标记 EXPERIMENTAL
