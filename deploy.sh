#!/usr/bin/env bash
# ==================================================================
# EvoRS-Comm Deployment Script
# ==================================================================
# Usage:
#   ./deploy.sh build     - Build Docker image
#   ./deploy.sh test      - Run tests in container
#   ./deploy.sh shell     - Open interactive shell in container
#   ./deploy.sh train     - Run Stage 2 training
#   ./deploy.sh dry-run   - Run Stage 1 dry run
#   ./deploy.sh pull      - Git pull and rebuild
# ==================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default service
SERVICE="${SERVICE:-ascend}"

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker image for Ascend NPU"
    echo "  build-cuda Build Docker image for NVIDIA CUDA"
    echo "  test      Run pytest inside container"
    echo "  shell     Open interactive shell"
    echo "  train     Run Stage 2 HM-MAGRPO training"
    echo "  dry-run   Run Stage 1 dry run"
    echo "  smoke     Quick smoke test (pytest + dry_run)"
    echo "  pull      Git pull + rebuild"
    echo "  logs      View container logs"
    echo "  stop      Stop container"
    echo ""
    echo "Environment:"
    echo "  SERVICE=ascend|cuda|cpu  (default: ascend)"
    exit 1
}

cmd_build() {
    echo "=== Building Ascend NPU image ==="
    docker compose build ascend
}

cmd_build_cuda() {
    echo "=== Building CUDA image ==="
    docker compose build cuda
}

cmd_test() {
    echo "=== Running tests in ${SERVICE} container ==="
    docker compose run --rm "${SERVICE}" python3 -m pytest -q --tb=short
}

cmd_shell() {
    echo "=== Opening shell in ${SERVICE} container ==="
    docker compose up -d "${SERVICE}"
    docker compose exec "${SERVICE}" bash
}

cmd_train() {
    echo "=== Running Stage 2 training in ${SERVICE} container ==="
    docker compose up -d "${SERVICE}"
    docker compose exec "${SERVICE}" python3 scripts/train_stage2.py \
        --config configs/experiments/stage2_pilot.yaml \
        --output outputs/stage2
}

cmd_dry_run() {
    echo "=== Running Stage 1 dry run in ${SERVICE} container ==="
    docker compose up -d "${SERVICE}"
    docker compose exec "${SERVICE}" python3 scripts/dry_run_stage1.py
}

cmd_smoke() {
    echo "=== Smoke test in ${SERVICE} container ==="
    docker compose up -d "${SERVICE}"
    docker compose exec "${SERVICE}" bash -c "
        echo '--- pytest ---'
        python3 -m pytest -q --tb=short
        echo ''
        echo '--- dry_run_stage1 ---'
        python3 scripts/dry_run_stage1.py
        echo ''
        echo '--- check_device ---'
        python3 scripts/check_device.py --device auto
        echo ''
        echo '=== SMOKE TEST PASSED ==='
    "
}

cmd_pull() {
    echo "=== Git pull + rebuild ==="
    git pull --ff-only
    docker compose build "${SERVICE}"
    echo "=== Done. Restart container to apply changes. ==="
    docker compose up -d "${SERVICE}"
}

cmd_logs() {
    docker compose logs -f "${SERVICE}"
}

cmd_stop() {
    docker compose stop "${SERVICE}"
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
    *)          usage ;;
esac
