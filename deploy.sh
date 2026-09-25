#!/usr/bin/env bash
# ==================================================================
# EvoRS-Comm Deployment Script
# ==================================================================
# Hot-mount workflow:
#   1. Build image ONCE:       ./deploy.sh build
#   2. After code changes:     git pull (no rebuild needed!)
#      Code is volume-mounted, changes are reflected immediately.
#   3. Only rebuild when:      pyproject.toml / requirements / Dockerfile change
# ==================================================================
# Usage:
#   ./deploy.sh build     - Build Docker image (first time or deps change)
#   ./deploy.sh test      - Run tests in container
#   ./deploy.sh shell     - Open interactive shell in container
#   ./deploy.sh train     - Train the Phase 5 routing Controller
#   ./deploy.sh dry-run   - Verify the Phase 5 checkpoint and host contract
#   ./deploy.sh pull      - Git pull (auto-detects if rebuild needed)
# ==================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default service
SERVICE="${SERVICE:-cpu}"
COMPOSE_ENV_FILE="${COMPOSE_ENV_FILE:-configs/docker/compose.defaults.env}"

compose() {
    docker compose --env-file "${COMPOSE_ENV_FILE}" "$@"
}

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker image for SERVICE"
    echo "  build-cuda  Build Docker image for NVIDIA CUDA"
    echo "  test        Run pytest inside container"
    echo "  shell       Open interactive shell"
    echo "  train       Train the Phase 5 routing Controller"
    echo "  dry-run     Verify the Phase 5 checkpoint and host contract"
    echo "  smoke       Run tests + Phase 5 checkpoint validation"
    echo "  pull        Git pull (no rebuild if code-only change)"
    echo "  logs        View container logs"
    echo "  stop        Stop container"
    echo "  deploy-npu  Create NPU deployment package (tar.gz)"
    echo "  sync-npu    rsync code to NPU server (set NPU_HOST env)"
    echo "  upload-data Show OBS data upload instructions"
    echo ""
    echo "Environment:"
    echo "  SERVICE=ascend|cuda|cpu  (default: cpu)"
    echo "  NPU_HOST=user@npu-server  (for sync-npu)"
    exit 1
}

cmd_build() {
    echo "=== Building ${SERVICE} image ==="
    compose build "${SERVICE}"
}

cmd_build_cuda() {
    echo "=== Building CUDA image ==="
    compose build cuda
}

cmd_test() {
    echo "=== Running tests in ${SERVICE} container ==="
    compose run --rm "${SERVICE}" python3 -m pytest -q --tb=short
}

cmd_shell() {
    echo "=== Opening shell in ${SERVICE} container ==="
    compose run --rm "${SERVICE}" bash
}

cmd_train() {
    echo "=== Running Phase 5 routing-controller training in ${SERVICE} container ==="
    compose run --rm "${SERVICE}" python3 scripts/train_phase5_routing.py \
        --config configs/experiments/phase5_routing.yaml \
        --output evidence/phase5-routing-container
}

cmd_dry_run() {
    echo "=== Verifying Phase 5 checkpoint in ${SERVICE} container ==="
    compose run --rm "${SERVICE}" python3 scripts/train_npu.py verify \
        --checkpoint evidence/2026-09-03_phase5_routing/checkpoint.json \
        --output evidence/phase5-container-environment.json
}

cmd_smoke() {
    echo "=== Smoke test in ${SERVICE} container ==="
    compose run --rm "${SERVICE}" bash -c "
        echo '--- pytest ---'
        python3 -m pytest -q --tb=short
        echo ''
        echo '--- validate_phase5 ---'
        python3 scripts/validate_phase5.py \
          --evidence evidence/2026-09-03_phase5_routing
        echo ''
        echo '--- verify checkpoint/environment contract ---'
        python3 scripts/train_npu.py verify \
          --checkpoint evidence/2026-09-03_phase5_routing/checkpoint.json \
          --output evidence/phase5-container-environment.json
        echo ''
        echo '=== SMOKE TEST PASSED ==='
    "
}

cmd_pull() {
    echo "=== Git pull (hot-mount, no rebuild) ==="
    git pull --ff-only

    # Only rebuild if dependencies changed (not for code changes)
    if git diff HEAD@{1} --name-only 2>/dev/null | grep -qE '(pyproject\.toml|requirements|Dockerfile|docker-compose)'; then
        echo "Dependencies/Dockerfile changed, rebuilding image..."
        compose build "${SERVICE}"
    else
        echo "Code-only change. No rebuild needed (hot-mount)."
    fi

    # Ensure container is running (no-op if already up)
    compose up -d "${SERVICE}"
    echo "=== Done. ==="
}

cmd_logs() {
    compose logs -f "${SERVICE}"
}

cmd_stop() {
    compose stop "${SERVICE}"
}

cmd_deploy_npu() {
    echo "=== Preparing NPU deployment package ==="

    # Create deployment archive
    ARCHIVE="evors-comm-$(date +%Y%m%d-%H%M%S).tar.gz"
    echo "Creating ${ARCHIVE}..."

    tar czf "${ARCHIVE}" \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='outputs' \
        --exclude='data' \
        --exclude='.git' \
        --exclude='*.pyc' \
        src/ tests/ scripts/ configs/ \
        pyproject.toml requirements-*.txt \
        docker-compose.npu.yml docker/Dockerfile.ascend \
        deploy.sh

    echo "Archive: ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"
    echo ""
    echo "To deploy to NPU server:"
    echo "  scp ${ARCHIVE} user@npu-server:/opt/"
    echo "  ssh user@npu-server 'cd /opt && tar xzf ${ARCHIVE} && cd evors-comm && docker compose --env-file configs/docker/compose.defaults.env -f docker-compose.npu.yml up -d'"
}

cmd_sync_to_npu() {
    echo "=== Syncing code to NPU server ==="
    NPU_HOST="${NPU_HOST:-}"
    NPU_USER="${NPU_USER:-root}"

    if [ -z "${NPU_HOST}" ]; then
        echo "Set NPU_HOST environment variable:"
        echo "  export NPU_HOST=user@npu-server"
        exit 1
    fi

    rsync -avz --progress \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='outputs' \
        --exclude='data' \
        --exclude='.git' \
        --exclude='*.pyc' \
        ./ "${NPU_HOST}:/opt/evors-comm/"

    echo ""
    echo "Code synced. To start:"
    echo "  ssh ${NPU_HOST} 'cd /opt/evors-comm && docker compose --env-file configs/docker/compose.defaults.env -f docker-compose.npu.yml up -d'"
}

cmd_upload_data() {
    echo "=== Uploading data to OBS ==="
    echo "Set OBS credentials first:"
    echo "  export OBS_ENDPOINT=obs.cn-north-4.myhuaweicloud.com"
    echo "  export OBS_BUCKET=evors-data"
    echo "  export OBS_ACCESS_KEY_ID=..."
    echo "  export OBS_SECRET_ACCESS_KEY=..."
    echo ""
    echo "Then use the download_data.py script on the prep server:"
    echo "  python scripts/download_data.py --all"
}

# Main
case "${1:-}" in
    build)      cmd_build ;;
    build-cuda) cmd_build_cuda ;;
    test)       cmd_test ;;
    shell)      cmd_shell ;;
    train)      cmd_train ;;
    dry-run)    cmd_dry_run ;;
    smoke)      cmd_smoke ;;
    pull)       cmd_pull ;;
    logs)       cmd_logs ;;
    stop)       cmd_stop ;;
    deploy-npu) cmd_deploy_npu ;;
    sync-npu)   cmd_sync_to_npu ;;
    upload-data) cmd_upload_data ;;
    *)          usage ;;
esac
