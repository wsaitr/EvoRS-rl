#!/usr/bin/env bash
# ==================================================================
# One-time setup for Ascend 910B NPU server
# ==================================================================
# Run this on the GPU server to set up the environment.
# Assumes: Ubuntu 22.04/24.04, ARM64, root access
# ==================================================================
set -euo pipefail

echo "=== EvoRS-Comm NPU Server Setup ==="

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git docker.io docker-compose-v2

# 2. Enable Docker
echo "[2/6] Configuring Docker..."
systemctl enable docker
systemctl start docker

# 3. Create project directory
echo "[3/6] Setting up project..."
mkdir -p /opt/evors-comm/{data,outputs,logs}

# 4. Clone repo
echo "[4/6] Cloning repository..."
cd /opt/evors-comm
if [ ! -d ".git" ]; then
    git clone https://github.com/wsaitr/EvoRS-rl.git .
fi

# 5. Create Python venv (for non-Docker use)
echo "[5/6] Creating Python venv..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev,data]"

# 6. Verify
echo "[6/6] Verifying..."
echo ""
echo "Python:"
.venv/bin/python --version
echo ""
echo "Docker:"
docker --version
docker compose version
echo ""
echo "NPU:"
npu-smi info 2>/dev/null | head -10 || echo "npu-smi not found (check CANN installation)"
echo ""
echo "PyTorch NPU:"
.venv/bin/python -c "import torch; import torch_npu; print(f'NPU available: {torch.npu.is_available()}, Devices: {torch.npu.device_count()}')" 2>/dev/null || echo "torch_npu not installed"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Set OBS credentials in environment"
echo "  2. Download data: .venv/bin/python scripts/download_data.py --all"
echo "  3. Build Docker: docker compose -f docker-compose.npu.yml build"
echo "  4. Run training: .venv/bin/python scripts/train_npu.py --config configs/experiments/stage2_pilot.yaml"
