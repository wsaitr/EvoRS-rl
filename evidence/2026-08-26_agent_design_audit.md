# Agent 设计审查结论（自查 + codex 交叉验证）

日期: 2026-08-26
方法: 隔离实验（同图同模型不同 prompt）+ codex read-only 源码审查交叉验证

## 一、单 agent 为什么比 direct 差（隔离实验实锤）

同一张图 (GT=2)、同一个 Qwen3.8-27B-Q4：
- direct 式 prompt（简短指令）×3 次：答 "2, 3, 2" —— 基本正确
- agent 协议 prompt（JSON 协议+工具说明+证据规则）：第一轮选择盲目 crop，
  第二轮同时看到整图+crop 后答 "1" —— 只数了 crop 里的

三个设计错误：
1. **无检测能力支撑的盲 crop**：VLM 没有定位能力却被要求决定裁哪里，
   裁剪位置随机，信息变差
2. **双图输入混淆**：整图+crop 同时给，"不要重复计数"规则让模型困惑，
   干脆只数 crop
3. **协议负担**：4-bit 量化模型输出严格 JSON 的能力有限，
   协议 token 挤占了视觉推理能力

## 二、codex 审查的确认发现（已核实）

### 确认为真的设计问题
1. ✅ reasoning expert 从不看检测到的 ROI 图，只看 bbox 文本 ——
   "层次协作"实际是 direct QA + 自生成文本前缀
2. ✅ 检测失败 fallback 创建全图 OBJECT 节点，reasoning 把它当真实检测 ——
   存在性任务正向偏置、计数任务锚定效应
3. ✅ latent L2 norm 恒为 1（编码器已归一化），cosine 未校准 ——
   注入 prompt 的数字无信息量（与 noise ablation 结论互证）
4. ✅ verifier 解析失败仍标 VERIFIED —— "验证器失效也通过"
5. ✅ FixedWorkflowPolicy 只有 visual→reasoning→verification 三专家，
   knowledge 专家在主臂中从不执行；knowledge 的 evidence.summary
   即使执行也不进 reasoning prompt
6. ✅ nested crop 坐标帧错误：对 crop 再裁剪时 source_bbox 记录局部坐标
7. ✅ 成本漏算：只有 reasoning 写 model_cost evidence，
   visual/verification/knowledge 的 token 全部没进统计
8. ✅ runtime 裸 except 吞专家异常 → NO_PROGRESS，错误被隐藏
9. ✅ arm 命名误导：hierarchical_comm 实际也是 full disclosure；
   flat_collab 也跑同样的空间检测 —— flat vs hierarchy 没有干净隔离

### codex 误报（已排除）
1. ❌ "AGENT_DISPATCHED 缺 call_id 会阻断运行" —— 实测 single agent
   正常完成（answer=2, completed）。emit() 内部有 call_id 注入逻辑，
   codex 只看了表面代码路径
2. ⚠️ GridCropTool overlap 公式"两列时没有 overlap" —— 部分真：
   2 列时中间列被边界对齐覆盖是几何必然（首尾列占满），
   3+ 列时 overlap 生效（有单测验证）

## 三、开源对照的核心差距

GeoChat/LISA/SkySenseGPT 都不是自主 tool-calling agent：
- GeoChat：单张整图 + 极短回答约束；ROI 来自用户显式给出，不盲 crop
- LISA：<SEG> token hidden state 经训练 MLP 进 SAM —— 端到端视觉接口
- SkySenseGPT：确定性 512x512 切片（训练时），非推理时动态裁剪
- RS-Agent / OpenEarthAgent（真正可借鉴）：
  * 任务路由在外部控制器
  * 感知交给专用工具（YOLOv8x-OBB 检测计数）
  * VLM 只做短视觉判断和语言聚合
  * 最终 VLM 看"带编号框的整图 + 实例表"，不给任意局部 crop

## 四、修复优先级（按影响排序）

P0（直接决定准确率）：
1. reasoning 不注入任何自生成文本/数字 —— 保持 clean prompt（已有消融证据）
2. verification 改为独立作答再程序比对（去 anchoring）
3. fallback 全图节点不再作为 detection 进入 reasoning 上下文

P1（实验有效性）：
4. arm 重命名 + flat 臂去掉 spatial 检测（真隔离 H1 变量）
5. 全部专家调用计入成本（visual/verification 也写 model_cost）
6. runtime 异常不再静默吞掉，记录到事件流

P2（论文口径）：
7. 2x2 消融分离 boxes/latent 各自贡献
8. 评测把异常计错并报告 coverage