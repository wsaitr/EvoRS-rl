#!/usr/bin/env bash
# ==================================================================
# Run EvoRS-Comm on GPU server (ModelArts Ascend 910B)
# ==================================================================
# Run on the GPU server
# Usage:
#   ./run_on_gpu.sh                          # Default: dry run
#   ./run_on_gpu.sh train                    # Start training
#   ./run_on_gpu.sh shell                    # Interactive shell
#   ./run_on_gpu.sh test                     # Run tests
# ==================================================================
set -euo pipefail

# PFS mount point (auto-mounted by ModelArts)
PFS_ROOT="${PFS_ROOT:-/lws2/evors-comm}"

# Docker image
IMAGE_NAME="${IMAGE_NAME:-evors-rl-npu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_PATH="${PFS_ROOT}/images/${IMAGE_NAME}-${IMAGE_TAG}.tar"

# NPU devices to expose
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (default)   Dry run with MockBackend"
    echo "  train       Start HM-MAGRPO training"
    echo "  test        Run pytest"
    echo "  shell       Interactive shell"
    echo "  load        Load Docker image from PFS"
    echo ""
    echo "Environment:"
    echo "  PFS_ROOT     PFS mount point (default: /lws2/evors-comm)"
    echo "  NPU_DEVICES  NPU devices to expose (default: 0,1,2,3,4,5,6,7)"
    exit 1
}

check_pfs() {
    if [ ! -d "${PFS_ROOT}/src" ]; then
        echo "ERROR: PFS not mounted or code not synced."
        echo "  PFS_ROOT=${PFS_ROOT}"
        echo "  Expected: ${PFS_ROOT}/src/ to exist"
        exit 1
    fi
}

load_image() {
    if [ ! -f "${IMAGE_PATH}" ]; then
        echo "ERROR: Docker image not found at ${IMAGE_PATH}"
        echo "Run sync_to_pfs.sh --image on the prep server first."
        exit 1
    fi

    echo "=== Loading Docker image ==="
    echo "Source: ${IMAGE_PATH} ($(du -h "${IMAGE_PATH}" | cut -f1))"
    docker load -i "${IMAGE_PATH}"
    echo "Image loaded: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# Common docker run args
docker_run_args() {
    echo "
        --rm
        -v ${PFS_ROOT}/src:/app/src:ro
        -v ${PFS_ROOT}/scripts:/app/scripts:ro
        -v ${PFS_ROOT}/configs:/app/configs:ro
        -v ${PFS_ROOT}/tests:/app/tests:ro
        -v ${PFS_ROOT}/data:/app/data:rw
        -v ${PFS_ROOT}/outputs:/app/outputs:rw
        -v ${PFS_ROOT}/logs:/app/logs:rw
        --shm-size=32g
        --network=host
        -e ASCEND_VISIBLE_DEVICES=${NPU_DEVICES}
        -e HCCL_WHITELIST_DISABLE=1
        -e OBS_ENDPOINT=${OBS_ENDPOINT:-}
        -e OBS_BUCKET=${OBS_BUCKET:-evors-data}
        -e OBS_ACCESS_KEY_ID=${OBS_ACCESS_KEY_ID:-}
        -e OBS_SECRET_ACCESS_KEY=${OBS_SECRET_ACCESS_KEY:-}
    "
}

# Build device mount list
device_args() {
    local args=""
    IFS=',' read -ra NPUS <<< "${NPU_DEVICES}"
    for i in "${NPUS[@]}"; do
        args="${args} --device=/dev/davinci${i}:/dev/davinci${i}"
    done
    args="${args} --device=/dev/davinci_manager:/dev/davinci_manager"
    args="${args} --device=/dev/devmm_svm:/dev/devmm_svm"
    args="${args} --device=/dev/hisi_hdc:/dev/hisi_hdc"
    echo "${args}"
}

cmd_dry_run() {
    check_pfs
    load_image

    echo "=== Running dry run (MockBackend) ==="
    docker run $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 scripts/dry_run_stage1.py
}

cmd_train() {
    check_pfs
    load_image

    CONFIG="${1:-configs/experiments/stage2_pilot.yaml}"

    echo "=== Starting HM-MAGRPO training ==="
    echo "Config: ${CONFIG}"
    echo "NPUs: ${NPU_DEVICES}"

    docker run $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 scripts/train_npu.py \
        --config "${CONFIG}" \
        --distributed \
        --output /app/outputs/training
}

cmd_test() {
    check_pfs
    load_image

    echo "=== Running tests ==="
    docker run $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 -m pytest tests/ -q --tb=short
}

cmd_shell() {
    check_pfs
    load_image

    echo "=== Opening shell ==="
    docker run -it $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash
}

# Main
case "${1:-dry-run}" in
    dry-run)    cmd_dry_run ;;
    train)      cmd_train "${2:-}" ;;
    test)       cmd_test ;;
    shell)      cmd_shell ;;
    load)       load_image ;;
    *)          usage ;;
esac
