#!/usr/bin/env bash
# ==================================================================
# setup_gpu_venv.sh — 轻量级 GPU 虚拟环境设置（无 Docker 方案）
# ==================================================================
# 适用于华为云 ModelArts bm.npu.arm.1snt9b1 (ARM64 + Ascend 910B)
# 服务器已预装：Python 3.11, PyTorch 2.9.0, CANN 8.5.1, verl 0.8.0
#
# 本脚本在预装 Python 之上创建一个轻量 venv，只补装项目所需的
# 轻量依赖（pytest/PyYAML/numpy/Pillow/datasets/obs/rich/tqdm），
# 然后通过软链接复用系统级的 torch / torch_npu / CANN / verl，
# 避免重复安装庞大的二进制包。
#
# 用法：
#   bash scripts/setup_gpu_venv.sh              # 完整设置（解压代码+创建venv+验证）
#   bash scripts/setup_gpu_venv.sh setup        # 同上
#   bash scripts/setup_gpu_venv.sh run <config> # 设置环境后直接启动训练
#   bash scripts/setup_gpu_venv.sh verify       # 仅验证已有环境
#   bash scripts/setup_gpu_venv.sh shell        # 进入激活的 venv shell
#   bash scripts/setup_gpu_venv.sh test         # 运行 pytest
#
# 环境变量：
#   WORK_DIR        工作目录（默认 /home/ma-user/evors-comm）
#   PIP_MIRROR      pip 镜像源（默认阿里云）
#   OBS_TARBALL     OBS 上的代码压缩包路径
#   PFS_MOUNT       lws2 并行文件系统挂载点（若已自动挂载则自动探测）
#   NPU_DEVICES     可见 NPU 设备列表（默认 0,1,2,3,4,5,6,7）
#   PYTHON_BIN      系统 Python 路径（默认自动探测 python3.11）
# ==================================================================
set -euo pipefail

# -------------------- 默认参数 --------------------
WORK_DIR="${WORK_DIR:-/home/ma-user/evors-comm}"
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
OBS_TARBALL="${OBS_TARBALL:-obs://lws2/evors-comm/evors-comm.tar.gz}"
PFS_MOUNT="${PFS_MOUNT:-}"          # 留空则自动探测
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-}"        # 留空则自动探测
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_npu.py}"

# venv 路径
VENV_DIR="${WORK_DIR}/.venv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*" >&2; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

# -------------------- 辅助函数 --------------------

usage() {
    cat <<EOF
${BOLD}EvoRS-Comm GPU 环境设置脚本（无 Docker 方案）${NC}

${BOLD}用法:${NC}
  bash $0 [命令] [参数]

${BOLD}命令:${NC}
  setup (默认)     创建 venv，解压代码，安装依赖，验证环境
  run <config>     设置后直接启动训练（如 configs/experiments/stage2_pilot.yaml）
  verify           仅验证环境（import torch / torch_npu / verl）
  shell            激活 venv 并进入交互式 bash
  test             运行 pytest 测试套件

${BOLD}环境变量:${NC}
  WORK_DIR         工作目录，默认 /home/ma-user/evors-comm
  PIP_MIRROR       pip 镜像源，默认 ${PIP_MIRROR}
  OBS_TARBALL      OBS 代码包路径，默认 obs://lws2/evors-comm/evors-comm.tar.gz
  PFS_MOUNT        lws2 挂载点（留空自动探测）
  NPU_DEVICES      可见 NPU，默认 0,1,2,3,4,5,6,7
  PYTHON_BIN       系统 Python 路径（留空自动探测 python3.11）

${BOLD}示例:${NC}
  # 完整设置
  bash scripts/setup_gpu_venv.sh

  # 设置后立刻开始训练
  bash scripts/setup_gpu_venv.sh run configs/experiments/stage2_pilot.yaml

  # 只验证已有环境
  bash scripts/setup_gpu_venv.sh verify
EOF
    exit 1
}

# 探测 lws2 挂载点
detect_pfs_mount() {
    if [[ -n "${PFS_MOUNT}" ]]; then
        return
    fi

    # 尝试从 mount 信息中查找 lws2 / sfs 类型的挂载
    local mount_point
    mount_point=$(mount 2>/dev/null \
        | grep -Ei 'lws2|sfs|obsfs|s3fs' \
        | awk '{print $3}' \
        | head -1 || true)

    if [[ -z "${mount_point}" ]]; then
        # 回退到常见路径
        for candidate in /mnt/lws2 /lws2 /home/ma-user/lws2; do
            if [[ -d "${candidate}/evors-comm" ]]; then
                mount_point="${candidate}"
                break
            fi
        done
    fi

    if [[ -n "${mount_point}" ]]; then
        PFS_MOUNT="${mount_point}"
        log "探测到 PFS 挂载点: ${PFS_MOUNT}"
    else
        warn "未探测到 lws2 PFS 挂载点，将从 OBS tarball 解压代码"
    fi
}

# 探测系统 Python
detect_python() {
    if [[ -n "${PYTHON_BIN}" ]]; then
        return
    fi

    for candidate in python3.11 python3 python; do
        if command -v "${candidate}" &>/dev/null; then
            PYTHON_BIN="$(command -v "${candidate}")"
            break
        fi
    done

    if [[ -z "${PYTHON_BIN}" ]]; then
        err "未找到 Python，请设置 PYTHON_BIN 环境变量"
        exit 1
    fi

    local pyver
    pyver=$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log "使用 Python: ${PYTHON_BIN} (版本 ${pyver})"
}

# -------------------- 核心步骤 --------------------

step_extract_code() {
    info "===== 1/5 解压代码 ====="

    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    # 优先从 PFS 挂载点读取
    local pfs_code=""
    if [[ -n "${PFS_MOUNT}" && -f "${PFS_MOUNT}/evors-comm/evors-comm.tar.gz" ]]; then
        pfs_code="${PFS_MOUNT}/evors-comm/evors-comm.tar.gz"
    elif [[ -n "${PFS_MOUNT}" && -d "${PFS_MOUNT}/evors-comm/src" ]]; then
        # PFS 上已经是解压后的目录，直接用
        log "PFS 上已有源码目录: ${PFS_MOUNT}/evors-comm/src"
        # 如果 WORK_DIR 和 PFS 路径不同，建软链
        if [[ "${WORK_DIR}" != "${PFS_MOUNT}/evors-comm" ]]; then
            for d in src scripts configs tests docker pyproject.toml \
                     requirements-ascend.txt requirements-base.txt requirements-vlm.txt; do
                if [[ -e "${PFS_MOUNT}/evors-comm/${d}" && ! -e "${WORK_DIR}/${d}" ]]; then
                    ln -sf "${PFS_MOUNT}/evors-comm/${d}" "${WORK_DIR}/${d}"
                fi
            done
            log "已从 PFS 建立软链"
            return
        fi
        return
    fi

    if [[ -n "${pfs_code}" ]]; then
        log "从 PFS 读取代码包: ${pfs_code}"
        tar xzf "${pfs_code}" -C "${WORK_DIR}"
        log "代码已解压到 ${WORK_DIR}"
        return
    fi

    # 回退：从 OBS 下载
    local local_tar="/tmp/evors-comm.tar.gz"
    if [[ -f "${local_tar}" ]]; then
        warn "使用缓存的 tarball: ${local_tar}"
    else
        info "从 OBS 下载代码包: ${OBS_TARBALL}"
        if command -v obsutil &>/dev/null; then
            obsutil cp "${OBS_TARBALL}" "${local_tar}" -f
        elif command -v obsutil_linux_amd64_5.8.3 &>/dev/null; then
            obsutil_linux_amd64_5.8.3 cp "${OBS_TARBALL}" "${local_tar}" -f
        else
            err "未找到 obsutil，请手动下载 ${OBS_TARBALL} 到 ${local_tar}"
            exit 1
        fi
    fi

    tar xzf "${local_tar}" -C "${WORK_DIR}"
    log "代码已解压到 ${WORK_DIR}"
}

step_create_venv() {
    info "===== 2/5 创建 Python 虚拟环境 ====="

    detect_python

    if [[ -d "${VENV_DIR}" ]]; then
        warn "venv 已存在: ${VENV_DIR}"
        info "如需重建，请先运行: rm -rf ${VENV_DIR}"
        return
    fi

    "${PYTHON_BIN}" -m venv "${VENV_DIR}" --system-site-packages
    log "venv 创建完成: ${VENV_DIR}"

    # 激活 venv（后续步骤在同一 shell 中执行）
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    # 升级 pip，设置镜像
    pip install --upgrade pip -q
    pip config set global.index-url "${PIP_MIRROR}" 2>/dev/null || true
    pip config set global.trusted-host "$(echo "${PIP_MIRROR}" | awk -F/ '{print $3}')" 2>/dev/null || true
    log "pip 镜像: ${PIP_MIRROR}"
}

step_install_deps() {
    info "===== 3/5 安装轻量依赖 ====="

    # 确保 venv 已激活
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    fi

    # 只安装轻量依赖 —— torch / torch_npu / CANN / verl 由系统 site-packages 提供
    # (--system-site-packages 创建时已包含)
    pip install --no-cache-dir \
        "pytest>=8" \
        "PyYAML>=6" \
        "numpy>=1.26" \
        "Pillow>=10" \
        "datasets>=2.19" \
        "esdk-obs-python>=3.0" \
        rich \
        tqdm

    log "轻量依赖安装完成"
}

step_setup_env() {
    info "===== 4/5 设置环境变量 ====="

    # 写出 sourceable 的环境文件，供 run / shell / 外部脚本使用
    local env_file="${WORK_DIR}/.env.gpu"
    cat > "${env_file}" <<ENVEOF
# EvoRS-Comm GPU 环境 — source 本文件以激活
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

# 激活 venv
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# 项目路径
export WORK_DIR="${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}/src\${PYTHONPATH:+:\${PYTHONPATH}}"

# NPU 设备
export ASCEND_VISIBLE_DEVICES="${NPU_DEVICES}"
export HCCL_WHITELIST_DISABLE=1

# CANN 环境（使用系统预装路径，若存在）
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.sh 2>/dev/null || true
fi
if [ -d /usr/local/Ascend/cann-toolkit/latest ]; then
    source /usr/local/Ascend/cann-toolkit/latest/bin/setenv.sh 2>/dev/null || true
fi

# 可选：OBS 凭证（通过外部环境变量注入）
# export OBS_ENDPOINT=...
# export OBS_BUCKET=evors-data
# export OBS_ACCESS_KEY_ID=...
# export OBS_SECRET_ACCESS_KEY=...
ENVEOF

    # 在当前 shell 中也导出关键变量
    export WORK_DIR
    export PYTHONPATH="${WORK_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export ASCEND_VISIBLE_DEVICES="${NPU_DEVICES}"
    export HCCL_WHITELIST_DISABLE=1

    log "环境变量文件: ${env_file}"
    info "其他终端使用: source ${env_file}"
}

step_verify() {
    info "===== 5/5 验证安装 ====="

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    fi

    cd "${WORK_DIR}"

    local failures=0
    local checks=(
        "torch:import torch; print(f'  torch {torch.__version__}  device={torch.cuda.is_available() and \"cuda\" or \"cpu\"}')"
        "torch_npu:import torch_npu; print(f'  torch_npu {torch_npu.__version__}  NPU可用: {torch_npu.npu.is_available()}')"
        "verl:import verl; print(f'  verl {getattr(verl, \"__version__\", \"OK\")}')"
        "numpy:import numpy; print(f'  numpy {numpy.__version__}')"
        "yaml:import yaml; print(f'  PyYAML {yaml.__version__}')"
        "PIL:import PIL; print(f'  Pillow {PIL.__version__}')"
        "datasets:import datasets; print(f'  datasets {datasets.__version__}')"
        "obs:import obs; print('  esdk-obs-python OK')"
        "rich:import rich; print(f'  rich {rich.__version__}')"
        "tqdm:import tqdm; print(f'  tqdm {tqdm.__version__}')"
        "pytest:import pytest; print(f'  pytest {pytest.__version__}')"
    )

    echo ""
    echo -e "${BOLD}========== 环境验证 ==========${NC}"

    for check in "${checks[@]}"; do
        local pkg="${check%%:*}"
        local code="${check#*:}"
        if python -c "${code}" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} ${pkg}"
        else
            echo -e "  ${RED}✗${NC} ${pkg}  —  导入失败"
            ((failures++)) || true
        fi
    done

    # NPU 可用性汇总
    echo ""
    python -c "
import torch, torch_npu
npu_count = torch_npu.npu.device_count() if hasattr(torch_npu, 'npu') else 0
print(f'NPU 设备数量: {npu_count}')
for i in range(npu_count):
    name = torch_npu.npu.get_device_name(i) if hasattr(torch_npu.npu, 'get_device_name') else 'Ascend'
    print(f'  [{i}] {name}')
" 2>/dev/null || warn "无法获取 NPU 设备信息（可能不在 NPU 机器上）"

    echo -e "${BOLD}==============================${NC}"
    echo ""

    if [[ ${failures} -gt 0 ]]; then
        err "有 ${failures} 个包导入失败，请检查环境"
        return 1
    else
        log "所有依赖验证通过 ✓"
    fi
}

# -------------------- 命令 --------------------

cmd_setup() {
    echo -e "${BOLD}${CYAN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║   EvoRS-Comm GPU 环境设置（无Docker）    ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"

    detect_pfs_mount
    step_extract_code
    step_create_venv
    step_install_deps
    step_setup_env
    step_verify

    echo ""
    log "设置完成！"
    echo ""
    echo "下一步："
    echo -e "  ${CYAN}source ${WORK_DIR}/.env.gpu${NC}              # 激活环境"
    echo -e "  ${CYAN}cd ${WORK_DIR}${NC}"
    echo -e "  ${CYAN}python scripts/train_npu.py --config <cfg>${NC}  # 开始训练"
    echo ""
    echo "或者一键运行："
    echo -e "  ${CYAN}bash $0 run configs/experiments/stage2_pilot.yaml${NC}"
    echo ""
}

cmd_run() {
    local config="${1:-}"
    if [[ -z "${config}" ]]; then
        err "请指定训练配置文件"
        echo "用法: $0 run <config.yaml>"
        echo "示例: $0 run configs/experiments/stage2_pilot.yaml"
        exit 1
    fi

    # 如果 venv 不存在，先走完整 setup
    if [[ ! -d "${VENV_DIR}" ]]; then
        warn "venv 不存在，先执行完整 setup..."
        cmd_setup
    fi

    # 激活环境
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    export WORK_DIR
    export PYTHONPATH="${WORK_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export ASCEND_VISIBLE_DEVICES="${NPU_DEVICES}"
    export HCCL_WHITELIST_DISABLE=1

    # 尝试加载 CANN 环境
    for envscript in \
        /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.sh \
        /usr/local/Ascend/cann-toolkit/latest/bin/setenv.sh; do
        if [[ -f "${envscript}" ]]; then
            # shellcheck disable=SC1090
            source "${envscript}" 2>/dev/null || true
            break
        fi
    done

    cd "${WORK_DIR}"

    echo -e "${BOLD}${GREEN}===== 开始训练 =====${NC}"
    echo "配置: ${config}"
    echo "NPU:  ${NPU_DEVICES}"
    echo "PWD:  $(pwd)"
    echo ""

    python "${TRAIN_SCRIPT}" \
        --config "${config}" \
        --distributed \
        --output "${WORK_DIR}/outputs/training"
}

cmd_verify_only() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        err "venv 不存在: ${VENV_DIR}"
        echo "请先运行: bash $0 setup"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    export PYTHONPATH="${WORK_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    step_verify
}

cmd_shell() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        err "venv 不存在: ${VENV_DIR}"
        echo "请先运行: bash $0 setup"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    export WORK_DIR
    export PYTHONPATH="${WORK_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export ASCEND_VISIBLE_DEVICES="${NPU_DEVICES}"
    export HCCL_WHITELIST_DISABLE=1

    cd "${WORK_DIR}"

    echo -e "${GREEN}EvoRS-Comm GPU 环境已激活${NC}"
    echo -e "  WORK_DIR=${WORK_DIR}"
    echo -e "  PYTHON=${VIRTUAL_ENV}/bin/python"
    echo -e "  NPU=${NPU_DEVICES}"
    echo ""
    exec bash --norc --noprofile -i
}

cmd_test() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        err "venv 不存在: ${VENV_DIR}"
        echo "请先运行: bash $0 setup"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    export PYTHONPATH="${WORK_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

    cd "${WORK_DIR}"
    python -m pytest tests/ -q --tb=short
}

# -------------------- 入口 --------------------
main() {
    local cmd="${1:-setup}"
    shift 2>/dev/null || true

    case "${cmd}" in
        setup)   cmd_setup ;;
        run)     cmd_run "$@" ;;
        verify)  cmd_verify_only ;;
        shell)   cmd_shell ;;
        test)    cmd_test ;;
        -h|--help|help) usage ;;
        *)
            err "未知命令: ${cmd}"
            usage
            ;;
    esac
}

main "$@"
