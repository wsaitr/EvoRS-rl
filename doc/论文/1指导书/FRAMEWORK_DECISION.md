# Framework decision

> 本文件原内容对应早期 RS-HMComm 原型，已被目标架构 v2 取代。

当前架构决策的唯一入口：

- [EvoRS-Comm 核心架构决策 v2](./EVORS_COMM_ARCHITECTURE_V2.md)

当前已确定：

- LangGraph 作为多 Agent runtime；
- `EvoRSController + RoutingPolicy` 负责跨 Agent 动态路由；
- 单 Agent 内部演化与跨 Agent 协作严格分离；
- 采用 Cognitive Agent Pool；
- 采用受控共享 Workspace + 显式消息路由；
- 采用问题驱动的轻量层次空间表示；
- Phase 5 的独立离散 `LearnedRoutingPolicy` 使用直接 on-policy policy gradient；
- `verl` 保留为未来 token-policy 的 AgentLoop/reward 适配，不把 dry-run 写成已训练；
- 旧版 own orchestration、CoMLRL/MAGRPO 优先路线和 MARTI 路线仅保留为历史实现参考。

硬件与第三方框架仍需通过 adapter 隔离，不能假设上游项目无需修改即可在 Ascend 上运行。
