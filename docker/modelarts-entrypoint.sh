#!/usr/bin/env bash
set -euo pipefail

evors_root="${EVORS_ROOT:-/opt/evors}"
output_root="${EVORS_OUTPUT_DIR:-/home/ma-user/evors-output}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "${output_root}"
cd "${evors_root}"

mode="${1:-verify}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${mode}" in
    verify)
        exec "${python_bin}" scripts/train_npu.py verify \
            --checkpoint "${evors_root}/checkpoints/phase5-routing.json" \
            --output "${output_root}/ascend-environment.json" \
            --require-npu \
            "$@"
        ;;
    routing)
        exec "${python_bin}" scripts/train_npu.py routing \
            --config "${evors_root}/configs/experiments/phase5_routing.yaml" \
            --output "${output_root}/phase5-routing" \
            "$@"
        ;;
    test)
        exec "${python_bin}" -m pytest -q --tb=short "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    *)
        exec "${mode}" "$@"
        ;;
esac
