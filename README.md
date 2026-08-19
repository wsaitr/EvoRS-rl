# RS-HMComm Starter

**Remote-Sensing Hierarchical Multimodal Communication for Multi-Agent RL**

三阶段研究代码骨架：

1. **Stage 1：多 Agent 推理 + 动态通信**
   - 共享显式状态：`Scene -> Region -> Group -> Object`
   - Text + Structured Message
   - 可选 Visual Latent handle
   - 动态 Agent 激活与通信
2. **Stage 2：Structured-Communication MARL**
   - 优先接 CoMLRL / MAGRPO
   - 通信 action 包括 recipient / level / modality / payload
   - reward 包括 task、communication cost、redundancy、structure/evidence
3. **Stage 3：完整实验**
   - 主训练 VRSBench
   - 外部评测 CHOICE、XLRS-Bench、GEO-Bench-VLM
   - matched-budget 对照和消融

## 为什么不直接 fork CoMLRL / MARTI

共享状态、通信协议、Agent、数据集和硬件后端先做成独立层：

- Stage 1 不依赖 RL；
- CUDA 与 Ascend 共享同一业务逻辑；
- Stage 2 通过 adapter 接 CoMLRL；
- MARTI 作为第二实现和大规模异构训练候选。

## 快速 dry run

```bash
cd rs_hmcomm_starter
python -m pip install -e .
python scripts/dry_run_stage1.py
pytest -q
```

dry run 只使用 MockBackend，不需要 GPU。

## 推荐第一版真实 VLM

先用 **Qwen2.5-VL-7B-Instruct**。

- NVIDIA：Transformers 直连做功能验证；后续可用 vLLM 做高吞吐 rollout。
- Ascend：CANN + torch_npu；高吞吐推理可用 vllm-ascend；需要中间 hidden state 时优先 Transformers + torch_npu。

## 重要限制

第一版先实现：
- text channel：✅
- structured tree channel：✅
- bbox/image crop reference：✅
- 直接 visual hidden-state 注入：实验性

真正 latent communication 与“把 crop 再发给另一个 Agent”不是一回事。
vLLM 服务路径通常不适合任意中间 hidden-state 注入，因此 latent channel 必须单独做 backend 适配。

详细见 `GUIDE.md` 和 `AI_IMPLEMENTATION_PROMPT.md`。
