#!/usr/bin/env bash
# ==================================================================
# Build Docker image ON the GPU server (native ARM64, fast)
# ==================================================================
# Run this on the ModelArts GPU server.
# It extracts the code from OBS, builds the thin Docker image,
# and saves it for reuse.
#
# Prerequisites:
#   - Code uploaded to obs://lws2/evors-comm/ (via sync_to_pfs.sh)
#   - Docker installed on the GPU server
#   - lws2 auto-mounted (check with: mount | grep lws2)
# ==================================================================
set -euo pipefail

# Where lws2 is mounted on the GPU server
PFS_MOUNT="${PFS_MOUNT:-/mnt/lws2}"
CODE_DIR="${PFS_MOUNT}/evors-comm"
IMAGE_NAME="${IMAGE_NAME:-evors-rl-gpu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

usage() {
    echo "Usage: $0"
    echo ""
    echo "Builds the Docker image on the GPU server (native ARM64)."
    echo "Code should already be at: ${CODE_DIR}"
    echo ""
    echo "Environment:"
    echo "  PFS_MOUNT   Where lws2 is mounted (default: /mnt/lws2)"
    echo "  IMAGE_TAG   Docker image tag (default: latest)"
    exit 1
}

# Check PFS mount
if [ ! -d "${CODE_DIR}" ]; then
    echo "ERROR: Code not found at ${CODE_DIR}"
    echo ""
    echo "Options:"
    echo "  1. Check lws2 mount point: mount | grep -i sfs"
    echo "  2. Set PFS_MOUNT to the correct path"
    echo "  3. If code not yet uploaded, run on prep server:"
    echo "     bash scripts/sync_to_pfs.sh"
    echo ""
    echo "If lws2 is not mounted, extract from OBS tar:"
    echo "  mkdir -p ${CODE_DIR}"
    echo "  obsutil cp obs://lws2/evors-comm/evors-comm.tar.gz /tmp/"
    echo "  tar xzf /tmp/evors-comm.tar.gz -C ${CODE_DIR}"
    exit 1
fi

echo "=== Building Docker image on GPU server ==="
echo "Code dir: ${CODE_DIR}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Build natively (ARM64 on ARM64 = fast)
docker build \
    -f "${CODE_DIR}/docker/Dockerfile.gpu" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${CODE_DIR}"

echo ""
echo "=== Image built successfully ==="
docker images "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "=== Ready to run ==="
echo ""
echo "  # Dry run (MockBackend):"
echo "  docker run --rm \\"
echo "    -v ${CODE_DIR}/src:/app/src:ro \\"
echo "    -v ${CODE_DIR}/scripts:/app/scripts:ro \\"
echo "    -v ${CODE_DIR}/configs:/app/configs:ro \\"
echo "    -v ${CODE_DIR}/tests:/app/tests:ro \\"
echo "    --device=/dev/davinci0 \\"
echo "    --device=/dev/davinci_manager \\"
echo "    --device=/dev/devmm_svm \\"
echo "    --device=/dev/hisi_hdc \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    python3 scripts/dry_run_stage1.py"
echo ""
echo "  # Training:"
echo "  docker run --rm \\"
echo "    -v ${CODE_DIR}/src:/app/src:ro \\"
echo "    -v ${CODE_DIR}/scripts:/app/scripts:ro \\"
echo "    -v ${CODE_DIR}/configs:/app/configs:ro \\"
echo "    -v ${CODE_DIR}/data:/app/data:rw \\"
echo "    -v ${CODE_DIR}/outputs:/app/outputs:rw \\"
echo "    --device=/dev/davinci0 \\"
echo "    --device=/dev/davinci_manager \\"
echo "    --device=/dev/devmm_svm \\"
echo "    --device=/dev/hisi_hdc \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    python3 scripts/train_npu.py --config configs/experiments/stage2_pilot.yaml"
