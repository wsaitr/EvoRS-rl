# EvoRS-Comm 部署指南

## 架构

```
┌──────────────┐   push    ┌──────────┐   pull    ┌─────────────────┐
│  本地开发     │ ────────> │  GitHub   │ ────────> │  前置服务器       │
│  Windows 11  │           │  EvoRS-rl │           │  1.95.150.79    │
│              │           └──────────┘           │  Ubuntu 24.04   │
└──────────────┘                                  │  x86_64, no NPU │
                                                   └────────┬────────┘
                                                            │
                                                   数据上传到 OBS
                                                            │
                                                            v
                                                   ┌─────────────────┐
                                                   │  GPU 训练服务器   │
                                                   │  ModelArts       │
                                                   │  N×Ascend 910B   │
                                                   │  ARM, 192GB RAM  │
                                                   │  data from OBS   │
                                                   └─────────────────┘
```

## 环境说明

| 环境 | 用途 | 配置 |
|------|------|------|
| 本地 (Windows) | 开发, Git push | Python 3.11 conda |
| 前置服务器 | 代码验证, 数据准备 | Ubuntu 24.04, Docker, venv |
| GPU 服务器 | 正式训练 | Ascend 910B × N, CANN 8.0, ARM |

## 快速开始

### 1. 本地开发

```bash
# 使用 conda 环境
conda activate rs-hmcomm

# 开发 → 测试 → 提交
python -m pytest -q
git add . && git commit -m "feat: ..."
git push origin main
```

### 2. 前置服务器 (1.95.150.79)

```bash
# SSH 登录
ssh root@1.95.150.79

# 拉取最新代码
cd /root/EvoRS-rl && git pull

# 运行测试 (venv)
.venv/bin/python -m pytest -q

# 运行测试 (Docker)
docker compose run --rm cpu

# 准备数据上传到 OBS
.venv/bin/python scripts/download_data.py --all
```

### 3. GPU 服务器 (Ascend 910B) — 推荐方案（venv，无Docker）

GPU 服务器预装了 PyTorch 2.9.0 / CANN 8.5.1 / verl 0.8.0 / Python 3.11。
使用 venv（`--system-site-packages`）直接复用预装环境，只补装轻量依赖。

```bash
# ===== 一键设置 =====
# 代码已从 OBS (lws2) 自动解压，venv 自动创建
bash scripts/setup_gpu_venv.sh

# ===== 验证环境 =====
bash scripts/setup_gpu_venv.sh verify

# ===== 开始训练 =====
bash scripts/setup_gpu_venv.sh run configs/experiments/stage2_pilot.yaml

# ===== 其他命令 =====
bash scripts/setup_gpu_venv.sh shell    # 进入激活的 venv shell
bash scripts/setup_gpu_venv.sh test     # 运行 pytest
```

**GPU 启动流程（零构建）：**

```
lws2 并行文件系统               GPU 服务器
──────────────                ──────────────────
evors-comm/                    setup_gpu_venv.sh
├── src/          ──────────>  解压代码
├── configs/                   创建 venv (--system-site-packages)
├── scripts/                   pip install 轻量依赖
└── evors-comm.tar.gz          source .env.gpu
                               开始训练！
```

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `WORK_DIR` | `/home/ma-user/evors-comm` | 工作目录 |
| `PIP_MIRROR` | 阿里云 | pip 镜像源 |
| `PFS_MOUNT` | 自动探测 | lws2 挂载点 |
| `NPU_DEVICES` | `0,1,2,3,4,5,6,7` | 可见 NPU |

## 并行文件系统 (lws2) 部署流程

前置服务器和 GPU 服务器共享 lws2 并行文件系统（OBS POSIX 桶）。
代码通过 obsutil 上传到 `obs://lws2/evors-comm/`，GPU 服务器自动挂载后直接使用。

### 架构图

```
前置服务器 (x86)              lws2 并行文件系统          GPU 服务器 (ARM64)
──────────────              ──────────────────          ──────────────────
git pull                    /lws2/evors-comm/            setup_gpu_venv.sh
obsutil upload               ├── src/        ← 代码        ├── 解压代码
  ──────────────→            ├── configs/    ← 配置        ├── 创建 venv
                             ├── scripts/    ← 脚本        ├── pip install
                             ├── data/       ← 数据        └── 开始训练！
                             └── outputs/    ← 训练输出
```

### 前置服务器操作

```bash
# 同步代码到 OBS（GPU 服务器立即可见）
./scripts/sync_to_pfs.sh

# 完整初始化（代码 + 目录占位）
./scripts/sync_to_pfs.sh --full
```

### GPU 服务器操作

```bash
# 一键设置（自动从 PFS 解压代码 + 创建 venv）
bash scripts/setup_gpu_venv.sh

# 开始训练
bash scripts/setup_gpu_venv.sh run configs/experiments/stage2_pilot.yaml
```

### 日常更新流程

```bash
# 前置服务器: 更新代码并上传到 OBS
git pull
./scripts/sync_to_pfs.sh          # 代码立即对 GPU 服务器可见

# GPU 服务器: 代码已在 PFS 上，venv 不需要重建
# 如果代码结构变了（新增模块），只需重新解压：
rm -rf /home/ma-user/evors-comm/src
bash scripts/setup_gpu_venv.sh    # 重新解压 + 验证
```

### PFS 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PFS_ROOT` | `/lws2/evors-comm` | PFS 挂载点 |
| `NPU_DEVICES` | `0,1,2,3,4,5,6,7` | 暴露的 NPU 设备 |
| `IMAGE_NAME` | `evors-rl-npu` | Docker 镜像名 |
| `IMAGE_TAG` | `latest` | 镜像 tag |

## 数据管道

数据存储在华为云 OBS:
- Bucket: `evors-data`
- 结构: `vrsbench/`, `choice/`, `xlrs-bench/`, `geobench-vlm/`

```bash
# 配置 OBS 凭证
export OBS_ENDPOINT="obs.cn-north-4.myhuaweicloud.com"
export OBS_BUCKET="evors-data"
export OBS_ACCESS_KEY_ID="your-access-key"
export OBS_SECRET_ACCESS_KEY="your-secret-key"

# 下载数据 (所有数据集)
python scripts/download_data.py --all

# 下载指定数据集
python scripts/download_data.py --dataset vrsbench --split train --limit 500

# 列出可用数据集
python scripts/download_data.py --list
```

## Docker 服务

### 前置服务器

```bash
# CPU 模式 (验证代码)
docker compose build cpu
docker compose run --rm cpu          # 运行测试
docker compose run --rm cpu bash     # 交互模式
```

### GPU 服务器

```bash
# NPU 模式 (训练)
docker compose -f docker-compose.npu.yml up -d npu     # 启动常驻容器
docker compose -f docker-compose.npu.yml exec npu bash  # 进入容器
docker compose -f docker-compose.npu.yml run --rm train # 一次性训练任务
```

## 部署命令速查

| 命令 | 说明 |
|------|------|
| `deploy.sh build` | 构建前置服务器 Docker 镜像 |
| `deploy.sh test` | 运行测试 |
| `deploy.sh smoke` | 冒烟测试 |
| `deploy.sh deploy-npu` | 打包 NPU 部署包 |
| `deploy.sh sync-npu` | rsync 同步到 GPU 服务器 |
| `deploy.sh train` | 运行 Stage 2 训练 |
| `deploy.sh pull` | Git pull + rebuild |

## Ascend 910B 注意事项

1. **CANN 版本**: 使用 CANN 8.0, 对应 torch_npu 2.1.0
2. **精度**: 使用 float16 (910B 原生支持)
3. **分布式**: HCCL 后端, 支持 8 卡并行
4. **ARM 架构**: Docker 镜像使用 arm64 版本
5. **内存**: 每卡 32GB HBM, 训练时预留 4GB
6. **AMP**: 使用 O2 级别自动混合精度
7. **数据格式**: float16 优于 bfloat16 (910B 优化)

## GPU 时间优化建议

GPU 服务器时间宝贵，以下措施节省 GPU 时间:

1. **前置验证**: 所有代码改动先在前置服务器用 MockBackend 验证
2. **单元测试**: 40 个测试全部通过后再上 GPU
3. **数据预下载**: 数据提前下载到 OBS, GPU 服务器直接读取
4. **小规模 Pilot**: 先用 500 samples + 1 epoch 验证流程
5. **断点续训**: checkpoint 支持中断恢复
6. **多卡并行**: 使用 8 卡分布式训练加速
