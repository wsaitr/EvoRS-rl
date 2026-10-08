# 2026-10-08 论文初稿与阶段报告

- [中文阶段报告](../reports/EVORS_PHASE_REPORT_2026-10-08.md)：工作内容、设计理由、结果和后续任务。
- [PDF 论文初稿](../../output/pdf/EvoRS-Comm_Paper_Draft_2026-10-08.pdf)：方法、已保存实验结果、局限与正式评测方案。
- [LaTeX 源码](evors_comm_draft_2026-10-08.tex)：可继续修改，使用 XeLaTeX 和 Fandol 中文字体。

代码基线是 `530914e2d39847de29249cecf0dd1a663cea5f1c`。本文引用历史证据，生成文稿时没有重新执行模型或训练。已有 `doc/论文/1/` 文件保留；本版本单独存放，避免覆盖历史写作。

## 编译

在仓库根目录执行两次（第二次解析目录引用与文献编号）：

```powershell
New-Item -ItemType Directory -Force output/pdf | Out-Null
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf -jobname=EvoRS-Comm_Paper_Draft_2026-10-08 doc/paper/evors_comm_draft_2026-10-08.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf -jobname=EvoRS-Comm_Paper_Draft_2026-10-08 doc/paper/evors_comm_draft_2026-10-08.tex
```

发布 PDF 前需核对日志中的引用和溢出警告，将全部页面渲染检查。编译辅助文件不纳入 Git。
