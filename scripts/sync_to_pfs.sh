#!/usr/bin/env bash
# ==================================================================
# Sync code and Docker image to lws2 parallel file system
# ==================================================================
# Run on the prep server (1.95.150.79)
# Usage:
#   ./scripts/sync_to_pfs.sh              # Sync code only
#   ./scripts/sync_to_pfs.sh --image      # Also build and save Docker image
#   ./scripts/sync_to_pfs.sh --full       # Full: code + image + data dirs
# ==================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# PFS mount point (shared between prep server and GPU server)
PFS_ROOT="${PFS_ROOT:-/lws2/evors-comm}"
IMAGE_NAME="${IMAGE_NAME:-evors-rl-npu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

usage() {
    echo "Usage: $0 [--image] [--full]"
    echo ""
    echo "Options:"
    echo "  (none)     Sync code to PFS only"
    echo "  --image    Also build ARM64 Docker image and save to PFS"
    echo "  --full     Full sync: code + image + create data directories"
    echo ""
    echo "Environment:"
    echo "  PFS_ROOT   PFS mount point (default: /lws2/evors-comm)"
    echo "  IMAGE_TAG  Docker image tag (default: latest)"
    exit 1
}

sync_code() {
    echo "=== Syncing code to PFS ==="
    echo "Source: ${PROJECT_DIR}"
    echo "Target: ${PFS_ROOT}"

    mkdir -p "${PFS_ROOT}"

    rsync -av --progress \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='outputs/' \
        --exclude='data/cache/' \
        --exclude='*.egg-info' \
        "${PROJECT_DIR}/src/" "${PFS_ROOT}/src/"

    rsync -av --progress \
        --exclude='__pycache__' \
        "${PROJECT_DIR}/scripts/" "${PFS_ROOT}/scripts/"

    rsync -av --progress "${PROJECT_DIR}/configs/" "${PFS_ROOT}/configs/"
    rsync -av --progress "${PROJECT_DIR}/tests/" "${PFS_ROOT}/tests/"

    # Copy project metadata
    cp "${PROJECT_DIR}/pyproject.toml" "${PFS_ROOT}/pyproject.toml"
    cp "${PROJECT_DIR}/requirements-base.txt" "${PFS_ROOT}/requirements-base.txt" 2>/dev/null || true
    cp "${PROJECT_DIR}/requirements-ascend.txt" "${PFS_ROOT}/requirements-ascend.txt" 2>/dev/null || true
    cp "${PROJECT_DIR}/docker-compose.npu.yml" "${PFS_ROOT}/docker-compose.npu.yml"
    cp "${PROJECT_DIR}/docker/Dockerfile.ascend" "${PFS_ROOT}/docker/Dockerfile.ascend" 2>/dev/null || \
        (mkdir -p "${PFS_ROOT}/docker" && cp "${PROJECT_DIR}/docker/Dockerfile.ascend" "${PFS_ROOT}/docker/Dockerfile.ascend")

    echo "Code synced to ${PFS_ROOT}"
}

build_and_save_image() {
    echo "=== Building ARM64 Docker image ==="

    # Build for ARM64 (GPU server architecture)
    # Using docker buildx for cross-platform build
    cd "${PROJECT_DIR}"

    if docker buildx inspect arm64builder &>/dev/null; then
        echo "Using existing buildx builder"
    else
        echo "Creating buildx builder for linux/arm64..."
        docker buildx create --name arm64builder --platform linux/arm64 --use
    fi

    docker buildx build \
        --platform linux/arm64 \
        -f docker/Dockerfile.ascend \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        --load \
        .

    echo "=== Saving Docker image to PFS ==="
    mkdir -p "${PFS_ROOT}/images"
    IMAGE_PATH="${PFS_ROOT}/images/${IMAGE_NAME}-${IMAGE_TAG}.tar"

    docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "${IMAGE_PATH}"

    echo "Image saved: ${IMAGE_PATH} ($(du -h "${IMAGE_PATH}" | cut -f1))"
    echo ""
    echo "GPU server can load with:"
    echo "  docker load -i ${IMAGE_PATH}"
}

create_data_dirs() {
    echo "=== Creating PFS directories ==="
    mkdir -p "${PFS_ROOT}/data/cache"
    mkdir -p "${PFS_ROOT}/data/vrsbench"
    mkdir -p "${PFS_ROOT}/data/choice"
    mkdir -p "${PFS_ROOT}/data/xlrs-bench"
    mkdir -p "${PFS_ROOT}/data/geobench-vlm"
    mkdir -p "${PFS_ROOT}/outputs"
    mkdir -p "${PFS_ROOT}/logs"
    echo "Directories created under ${PFS_ROOT}"
}

# Main
case "${1:-}" in
    --image)
        sync_code
        build_and_save_image
        ;;
    --full)
        sync_code
        build_and_save_image
        create_data_dirs
        ;;
    "")
        sync_code
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "=== Sync complete ==="
echo "PFS root: ${PFS_ROOT}"
du -sh "${PFS_ROOT}" 2>/dev/null || true
