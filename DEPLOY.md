# EvoRS-Comm 服务器部署指南

## 架构

```
本地开发机                    Git 仓库                    服务器 (Ascend NPU)
┌──────────────┐     push     ┌──────────┐    pull     ┌──────────────────┐
│  编辑代码     │ ──────────> │  GitHub   │ ─────────> │  docker compose  │
│  src/rs_hmcomm│             │  /GitLab  │            │  ┌──────────────┐ │
│  tests/       │             └──────────┘            │  │ 容器 (热挂载)  │ │
│  configs/     │                                     │  │  src/ <- volume│ │
└──────────────┘                                     │  └──────────────┘ │
                                                      └──────────────────┘
```

## 快速开始

### 1. 初始化 Git 仓库

```bash
# 本地
cd f:\project\yaogan_lunwen
git init
git add .
git commit -m "feat: initial EvoRS-Comm implementation"

# 添加远程仓库 (替换为你的仓库地址)
git remote add origin git@github.com:YOUR_USER/evors-comm.git
# 或者 GitLab:
# git remote add origin git@YOUR_GITLAB:YOUR_USER/evors-comm.git

git push -u origin main
```

### 2. 服务器端设置

```bash
# SSH 登录服务器
ssh user@your-server

# 克隆仓库
git clone git@github.com:YOUR_USER/evors-comm.git
cd evors-comm

# 构建镜像
docker compose build ascend

# 运行冒烟测试
SERVICE=ascend bash deploy.sh smoke
```

### 3. 日常开发流程

```bash
# 本地：编辑代码
# 本地：提交并推送
git add .
git commit -m "feat: ..."
git push

# 服务器：拉取 + 重建
SERVICE=ascend bash deploy.sh pull

# 服务器：跑测试
SERVICE=ascend bash deploy.sh test

# 服务器：训练
SERVICE=ascend bash deploy.sh train
```

## 代码热挂载

`docker-compose.yml` 通过 volume mount 将 `src/`, `tests/`, `scripts/`, `configs/` 挂载到容器内。
**代码修改后立即在容器中生效**，无需重建镜像。

仅以下情况需要 `docker compose build`：
- 修改了 `requirements-*.txt`
- 修改了 `pyproject.toml` 的依赖
- 修改了 `docker/Dockerfile.*`

## 命令速查

| 命令 | 说明 |
|------|------|
| `deploy.sh build` | 构建 Ascend NPU 镜像 |
| `deploy.sh build-cuda` | 构建 CUDA 镜像 |
| `deploy.sh test` | 运行测试 |
| `deploy.sh shell` | 进入容器交互 |
| `deploy.sh train` | 运行 Stage 2 训练 |
| `deploy.sh dry-run` | 运行 Stage 1 dry run |
| `deploy.sh smoke` | 冒烟测试 (pytest + dry_run + device check) |
| `deploy.sh pull` | Git pull + rebuild |
| `deploy.sh logs` | 查看容器日志 |
| `deploy.sh stop` | 停止容器 |

## Ascend NPU 注意事项

1. **CANN 版本**: Dockerfile 默认使用 CANN 8.0.RC3.beta1，请根据服务器实际版本调整
2. **torch_npu 版本**: 必须与 CANN 版本匹配，参考 https://gitee.com/ascend/pytorch
3. **设备透传**: docker-compose 已配置 `/dev/davinci*` 设备透传
4. **HCCL 分布式**: 多卡训练需设置 `ASCEND_VISIBLE_DEVICES`
5. **vLLM**: 高吞吐推理可用 `vllm-ascend`，需单独安装

## 目录结构

```
project/
├── docker/
│   ├── Dockerfile.ascend    # Ascend NPU 镜像
│   └── Dockerfile.cuda      # NVIDIA CUDA 镜像
├── docker-compose.yml       # 服务编排 (ascend/cuda/cpu)
├── deploy.sh                # 一键部署脚本
├── src/rs_hmcomm/           # [热挂载] 源代码
├── tests/                   # [热挂载] 测试
├── scripts/                 # [热挂载] 脚本
├── configs/                 # [热挂载] 配置
├── outputs/                 # [持久化] 训练输出
└── logs/                    # [持久化] 日志
```
