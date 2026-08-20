#!/usr/bin/env bash
# ==================================================================
# Sync code to lws2 OBS bucket (POSIX)
# ==================================================================
# Run on the prep server (1.95.150.79)
#
# lws2 is an OBS POSIX bucket, auto-mounted on the GPU server.
# This script packages code and uploads via obsutil.
#
# Usage:
#   ./scripts/sync_to_pfs.sh              # Upload code tar.gz
#   ./scripts/sync_to_pfs.sh --docker     # Also build & save Docker image
#   ./scripts/sync_to_pfs.sh --full       # Code + Docker + data dirs
# ==================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# OBS paths
OBSUTIL="${OBSUTIL:-/root/obsutil_linux_amd64_5.8.3/obsutil}"
OBS_BUCKET="${OBS_BUCKET:-obs://lws2}"
OBS_PREFIX="${OBS_PREFIX:-${OBS_BUCKET}/evors-comm}"

# Docker image
IMAGE_NAME="${IMAGE_NAME:-evors-rl-npu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

usage() {
    echo "Usage: $0 [--docker] [--full]"
    echo ""
    echo "Uploads code to OBS (lws2). GPU server auto-mounts the bucket."
    echo ""
    echo "Options:"
    echo "  (none)     Upload code archive only"
    echo "  --docker   Also build ARM64 Docker image and upload"
    echo "  --full     Full: code + Docker + create data dirs on OBS"
    echo ""
    echo "Environment:"
    echo "  OBSUTIL     obsutil path (default: /root/obsutil_linux_amd64_5.8.3/obsutil)"
    echo "  OBS_PREFIX  OBS target (default: obs://lws2/evors-comm)"
    exit 1
}

upload_code() {
    echo "=== Packaging code ==="
    local ARCHIVE="/tmp/evors-comm-$(date +%Y%m%d-%H%M%S).tar.gz"

    cd "${PROJECT_DIR}"
    tar czf "${ARCHIVE}" \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='outputs' \
        --exclude='data/cache' \
        --exclude='*.egg-info' \
        src/ scripts/ configs/ tests/ \
        pyproject.toml requirements-*.txt \
        docker-compose*.yml docker/ \
        deploy.sh DEPLOY.md CLAUDE.md

    local SIZE
    SIZE=$(du -h "${ARCHIVE}" | cut -f1)
    echo "Archive: ${ARCHIVE} (${SIZE})"

    echo "=== Uploading to ${OBS_PREFIX}/ ==="
    "${OBSUTIL}" cp "${ARCHIVE}" "${OBS_PREFIX}/evors-comm.tar.gz" -f
    echo "Uploaded: ${OBS_PREFIX}/evors-comm.tar.gz"

    # On GPU server (where lws2 is mounted at e.g. /mnt/lws2):
    #   cd /mnt/lws2/evors-comm && tar xzf evors-comm.tar.gz
}

build_and_upload_docker() {
    echo "=== Building ARM64 Docker image ==="
    cd "${PROJECT_DIR}"

    # Check if buildx is available
    if ! docker buildx version &>/dev/null; then
        echo "ERROR: docker buildx not available. Install first."
        exit 1
    fi

    # Create/use ARM64 builder
    if ! docker buildx inspect arm64builder &>/dev/null; then
        docker buildx create --name arm64builder --platform linux/arm64 --use
    else
        docker buildx use arm64builder
    fi

    docker buildx build \
        --platform linux/arm64 \
        -f docker/Dockerfile.ascend \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        --load \
        .

    echo "=== Saving Docker image ==="
    local IMAGE_TAR="/tmp/${IMAGE_NAME}-${IMAGE_TAG}.tar"
    docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "${IMAGE_TAR}"

    local SIZE
    SIZE=$(du -h "${IMAGE_TAR}" | cut -f1)
    echo "Image: ${IMAGE_TAR} (${SIZE})"

    echo "=== Uploading Docker image to ${OBS_PREFIX}/images/ ==="
    "${OBSUTIL}" cp "${IMAGE_TAR}" "${OBS_PREFIX}/images/${IMAGE_NAME}-${IMAGE_TAG}.tar" -f
    echo "Uploaded: ${OBS_PREFIX}/images/${IMAGE_NAME}-${IMAGE_TAG}.tar"
}

create_obs_dirs() {
    echo "=== Creating OBS directories ==="
    # OBS doesn't need explicit dir creation, but we upload placeholder files
    for dir in data/cache data/vrsbench data/choice outputs logs; do
        echo "placeholder" | "${OBSUTIL}" cp - "${OBS_PREFIX}/${dir}/.keep" -f 2>/dev/null || true
    done
    echo "Directories created under ${OBS_PREFIX}"
}

# Main
case "${1:-}" in
    --docker)
        upload_code
        build_and_upload_docker
        ;;
    --full)
        upload_code
        build_and_upload_docker
        create_obs_dirs
        ;;
    "")
        upload_code
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "=== Upload complete ==="
echo "OBS prefix: ${OBS_PREFIX}"
echo ""
echo "On GPU server (lws2 auto-mounted):"
echo "  cd <mount_point>/evors-comm"
echo "  tar xzf evors-comm.tar.gz"
