#!/usr/bin/env bash
# ==================================================================
# Run EvoRS-Comm on GPU server (ModelArts Ascend 910B)
# ==================================================================
# GPU server 零构建启动：从 OBS 加载 Docker 镜像和代码
#
# Usage:
#   ./run_on_gpu.sh                          # Default: dry run
#   ./run_on_gpu.sh train                    # Start training
#   ./run_on_gpu.sh train <config>           # Custom config
#   ./run_on_gpu.sh shell                    # Interactive shell
#   ./run_on_gpu.sh test                     # Run tests
#   ./run_on_gpu.sh load                     # Load Docker image only
#
# OBS 资源（由前置服务器上传）：
#   obs://lws2/evors-comm/evors-comm.tar.gz              # 代码
#   obs://lws2/evors-comm/images/evors-rl-gpu-latest.tar  # Docker 镜像
# ==================================================================
set -euo pipefail

# ---- 路径配置 ----
# lws2 并行文件系统挂载点（ModelArts 自动挂载）
PFS_MOUNT="${PFS_MOUNT:-}"
OBS_BASE="obs://lws2/evors-comm"
WORK_DIR="${WORK_DIR:-/home/ma-user/evors-comm}"

# Docker 镜像
IMAGE_NAME="${IMAGE_NAME:-evors-rl-gpu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# NPU 设备
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"

# obsutil 路径（GPU 服务器上可能不同）
OBSUTIL="${OBSUTIL:-obsutil}"

# ---- 颜色 ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*" >&2; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  dry-run (默认)   MockBackend 测试运行"
    echo "  train [config]   启动训练"
    echo "  test             运行 pytest"
    echo "  shell            交互式 shell"
    echo "  load             加载 Docker 镜像"
    echo "  setup            从 OBS 下载代码和镜像（首次运行）"
    exit 1
}

# ---- 探测 PFS 挂载点 ----
detect_pfs() {
    if [[ -n "${PFS_MOUNT}" ]]; then
        return
    fi

    local mp
    mp=$(mount 2>/dev/null | grep -Ei 'lws2|sfs|obsfs|s3fs' | awk '{print $3}' | head -1 || true)

    if [[ -z "${mp}" ]]; then
        for candidate in /mnt/lws2 /lws2 /home/ma-user/lws2; do
            if [[ -d "${candidate}" ]]; then
                mp="${candidate}"
                break
            fi
        done
    fi

    if [[ -n "${mp}" ]]; then
        PFS_MOUNT="${mp}"
        log "PFS 挂载点: ${PFS_MOUNT}"
    else
        warn "未探测到 PFS 挂载点，将从 OBS 直接下载"
    fi
}

# ---- 从 OBS 下载 ----
download_from_obs() {
    local target_dir="${1}"
    local remote_path="${2}"
    local local_path="${3}"

    mkdir -p "${target_dir}"

    if [[ -f "${local_path}" ]]; then
        log "已缓存: ${local_path}"
        return 0
    fi

    info "从 OBS 下载: ${remote_path}"
    if command -v "${OBSUTIL}" &>/dev/null; then
        "${OBSUTIL}" cp "${remote_path}" "${local_path}" -f
    else
        err "obsutil 未找到。请安装或设置 OBSUTIL 环境变量"
        exit 1
    fi
}

# ---- Setup: 下载代码 + 镜像 ----
cmd_setup() {
    echo -e "${BOLD}${CYAN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║   EvoRS-Comm GPU 环境设置               ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"

    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    # 1. 代码：优先从 PFS 读取，否则从 OBS 下载
    if [[ -n "${PFS_MOUNT}" && -f "${PFS_MOUNT}/evors-comm/evors-comm.tar.gz" ]]; then
        log "从 PFS 读取代码"
        cp "${PFS_MOUNT}/evors-comm/evors-comm.tar.gz" /tmp/
    else
        download_from_obs /tmp "obs://lws2/evors-comm/evors-comm.tar.gz" "/tmp/evors-comm.tar.gz"
    fi

    info "解压代码到 ${WORK_DIR}"
    tar xzf /tmp/evors-comm.tar.gz -C "${WORK_DIR}"
    log "代码就绪"

    # 2. Docker 镜像：优先从 PFS 读取，否则从 OBS 下载
    local image_tar="${WORK_DIR}/evors-rl-gpu-latest.tar"
    if [[ -n "${PFS_MOUNT}" && -f "${PFS_MOUNT}/evors-comm/images/evors-rl-gpu-latest.tar" ]]; then
        log "从 PFS 读取 Docker 镜像"
        cp "${PFS_MOUNT}/evors-comm/images/evors-rl-gpu-latest.tar" "${image_tar}"
    else
        download_from_obs "${WORK_DIR}" "obs://lws2/evors-comm/images/evors-rl-gpu-latest.tar" "${image_tar}"
    fi

    # 3. 加载 Docker 镜像
    info "加载 Docker 镜像..."
    docker load -i "${image_tar}"
    log "镜像已加载: ${IMAGE_NAME}:${IMAGE_TAG}"

    # 4. 创建必要目录
    mkdir -p "${WORK_DIR}/data" "${WORK_DIR}/outputs" "${WORK_DIR}/logs"

    echo ""
    log "设置完成！"
    echo ""
    echo "运行命令："
    echo -e "  ${CYAN}bash scripts/run_on_gpu.sh test${NC}           # 测试"
    echo -e "  ${CYAN}bash scripts/run_on_gpu.sh train${NC}          # 训练"
    echo ""
}

# ---- 加载 Docker 镜像 ----
load_image() {
    # 检查镜像是否已加载
    if docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &>/dev/null; then
        log "镜像已加载: ${IMAGE_NAME}:${IMAGE_TAG}"
        return 0
    fi

    detect_pfs

    local image_tar=""
    if [[ -n "${PFS_MOUNT}" && -f "${PFS_MOUNT}/evors-comm/images/evors-rl-gpu-latest.tar" ]]; then
        image_tar="${PFS_MOUNT}/evors-comm/images/evors-rl-gpu-latest.tar"
    elif [[ -f "${WORK_DIR}/evors-rl-gpu-latest.tar" ]]; then
        image_tar="${WORK_DIR}/evors-rl-gpu-latest.tar"
    else
        err "Docker 镜像未找到。请先运行: bash $0 setup"
        exit 1
    fi

    info "加载镜像: ${image_tar} ($(du -h "${image_tar}" | cut -f1))"
    docker load -i "${image_tar}"
    log "镜像已加载: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# ---- 检查代码目录 ----
check_code() {
    local code_dir="${1}"
    if [[ ! -d "${code_dir}/src" ]]; then
        err "代码未找到: ${code_dir}/src"
        echo "请先运行: bash $0 setup"
        exit 1
    fi
}

# ---- Docker run 参数 ----
docker_run_args() {
    local code_dir="${1:-${WORK_DIR}}"
    echo "
        --rm
        -v ${code_dir}/src:/app/src:ro
        -v ${code_dir}/scripts:/app/scripts:ro
        -v ${code_dir}/configs:/app/configs:ro
        -v ${code_dir}/tests:/app/tests:ro
        -v ${code_dir}/data:/app/data:rw
        -v ${code_dir}/outputs:/app/outputs:rw
        -v ${code_dir}/logs:/app/logs:rw
        --shm-size=32g
        --network=host
        -e ASCEND_VISIBLE_DEVICES=${NPU_DEVICES}
        -e HCCL_WHITELIST_DISABLE=1
        -e PYTHONPATH=/app/src
        -e OBS_ENDPOINT=${OBS_ENDPOINT:-}
        -e OBS_BUCKET=${OBS_BUCKET:-lws2}
        -e OBS_ACCESS_KEY_ID=${OBS_ACCESS_KEY_ID:-}
        -e OBS_SECRET_ACCESS_KEY=${OBS_SECRET_ACCESS_KEY:-}
    "
}

# ---- NPU 设备参数 ----
device_args() {
    local args=""
    IFS=',' read -ra NPUS <<< "${NPU_DEVICES}"
    for i in "${NPUS[@]}"; do
        args="${args} --device=/dev/davinci${i}:/dev/davinci${i}"
    done
    # 管理设备
    for dev in davinci_manager devmm_svm hisi_hdc; do
        if [[ -e "/dev/${dev}" ]]; then
            args="${args} --device=/dev/${dev}:/dev/${dev}"
        fi
    done
    echo "${args}"
}

# ---- 命令 ----
cmd_dry_run() {
    load_image
    check_code "${WORK_DIR}"
    info "Dry run (MockBackend)..."
    docker run $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 scripts/dry_run_stage1.py
}

cmd_train() {
    load_image
    check_code "${WORK_DIR}"
    local config="${1:-configs/experiments/stage2_pilot.yaml}"

    echo -e "${BOLD}${GREEN}===== 开始训练 =====${NC}"
    echo "配置: ${config}"
    echo "NPU:  ${NPU_DEVICES}"
    echo ""

    docker run $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 scripts/train_npu.py \
        --config "${config}" \
        --distributed \
        --output /app/outputs/training
}

cmd_test() {
    load_image
    check_code "${WORK_DIR}"
    info "运行测试..."
    docker run $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 -m pytest tests/ -q --tb=short
}

cmd_shell() {
    load_image
    check_code "${WORK_DIR}"
    info "打开 shell..."
    docker run -it $(docker_run_args) $(device_args) \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash
}

# ---- Main ----
main() {
    detect_pfs
    local cmd="${1:-dry-run}"
    shift 2>/dev/null || true

    case "${cmd}" in
        setup)     cmd_setup ;;
        dry-run)   cmd_dry_run ;;
        train)     cmd_train "$@" ;;
        test)      cmd_test ;;
        shell)     cmd_shell ;;
        load)      load_image ;;
        -h|--help) usage ;;
        *)         err "未知命令: ${cmd}"; usage ;;
    esac
}

main "$@"
