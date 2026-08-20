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

### 3. GPU 服务器 (Ascend 910B)

```bash
# 代码同步 (方式一: 从 GitHub pull)
cd /opt/evors-comm && git pull

# 代码同步 (方式二: 从前置服务器 rsync)
# 在前置服务器上:
rsync -avz /root/EvoRS-rl/ gpu-server:/opt/evors-comm/ \
    --exclude='__pycache__' --exclude='.git' --exclude='data' --exclude='outputs'

# 启动 Docker 容器
docker compose -f docker-compose.npu.yml up -d npu

# 进入容器
docker compose -f docker-compose.npu.yml exec npu bash

# 验证 NPU
python3 -c "import torch; import torch_npu; print(torch.npu.device_count())"

# 下载 OBS 数据
python3 scripts/download_data.py --all

# 运行训练 (单卡)
python3 scripts/train_npu.py --config configs/experiments/stage2_pilot.yaml

# 运行训练 (多卡分布式)
torchrun --nproc_per_node=8 scripts/train_npu.py \
    --config configs/experiments/stage2_pilot.yaml \
    --distributed --output outputs/distributed_training
```

## 并行文件系统 (lws2) 部署流程

前置服务器和 GPU 服务器共享 lws2 并行文件系统。代码和 Docker 镜像通过 PFS 传递，GPU 服务器不需要 git pull 或 docker build。

### 架构图

```
前置服务器 (x86)              lws2 并行文件系统          GPU 服务器 (ARM64)
──────────────              ──────────────────          ──────────────────
git pull                    /lws2/evors-comm/
docker buildx build          ├── src/        ← 代码
docker save → tar  ──────→   ├── images/     ← Docker tar
rsync code    ──────→        ├── configs/    ← 配置
                             ├── data/       ← 数据(OBS)
                             └── outputs/    ← 训练输出
                                                          docker load < tar
                                                          docker run -v /lws2/...
                                                          训练开始
```

### 前置服务器操作

```bash
# 1. 同步代码到 PFS
./scripts/sync_to_pfs.sh

# 2. 构建 ARM64 Docker 镜像并保存到 PFS (首次或依赖变更时)
./scripts/sync_to_pfs.sh --image

# 3. 完整初始化 (代码+镜像+目录)
./scripts/sync_to_pfs.sh --full
```

### GPU 服务器操作

```bash
# 加载镜像并运行 (从 PFS 读取)
./scripts/run_on_gpu.sh           # MockBackend dry run
./scripts/run_on_gpu.sh train     # 开始训练
./scripts/run_on_gpu.sh test      # 跑测试
./scripts/run_on_gpu.sh shell     # 交互式 shell

# 仅加载镜像 (已加载过的跳过)
./scripts/run_on_gpu.sh load
```

### 日常更新流程

```bash
# 前置服务器: 更新代码
git pull
./scripts/sync_to_pfs.sh          # 代码立即对 GPU 服务器可见

# GPU 服务器: 直接运行 (代码从 PFS 读取，无需 pull)
./scripts/run_on_gpu.sh train

# 只有依赖变了才需要重建镜像:
# 前置服务器:
./scripts/sync_to_pfs.sh --image  # 重新 build + save
# GPU 服务器:
./scripts/run_on_gpu.sh load      # 重新 load
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
