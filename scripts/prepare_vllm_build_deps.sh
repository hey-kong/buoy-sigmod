#!/usr/bin/env bash
# Prepare pinned source dependencies required to build the Buoy-vLLM fork.
#
# Usage:
#   ./scripts/prepare_vllm_build_deps.sh [deps-dir]
#   source [deps-dir]/vllm_build_env.sh
#
# The default deps-dir is .deps/vllm-build under the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="${1:-${REPO_ROOT}/.deps/vllm-build}"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "${dir}/.git" ]]; then
    echo "Updating ${dir}"
    git -C "${dir}" fetch --tags --prune origin
  else
    echo "Cloning ${url} -> ${dir}"
    git clone "${url}" "${dir}"
  fi
}

checkout_ref() {
  local dir="$1"
  local ref="$2"
  echo "Checking out ${ref} in ${dir}"
  git -C "${dir}" checkout "${ref}"
}

mkdir -p "${DEPS_DIR}"

FLASH_ATTN_DIR="${DEPS_DIR}/flash-attention"
CUTLASS_DIR="${DEPS_DIR}/cutlass"
FLASH_MLA_DIR="${DEPS_DIR}/FlashMLA"
TRITON_DIR="${DEPS_DIR}/triton"

clone_or_update "https://github.com/vllm-project/flash-attention.git" "${FLASH_ATTN_DIR}"
checkout_ref "${FLASH_ATTN_DIR}" "188be16520ceefdc625fdf71365585d2ee348fe2"
git -C "${FLASH_ATTN_DIR}" submodule update --init --recursive

clone_or_update "https://github.com/nvidia/cutlass.git" "${CUTLASS_DIR}"
checkout_ref "${CUTLASS_DIR}" "v4.2.1"

clone_or_update "https://github.com/vllm-project/FlashMLA.git" "${FLASH_MLA_DIR}"
checkout_ref "${FLASH_MLA_DIR}" "c2afa9cb93e674d5a9120a170a6da57b89267208"
git -C "${FLASH_MLA_DIR}" submodule update --init --recursive

clone_or_update "https://github.com/triton-lang/triton.git" "${TRITON_DIR}"
checkout_ref "${TRITON_DIR}" "v3.5.0"

ENV_FILE="${DEPS_DIR}/vllm_build_env.sh"
cat > "${ENV_FILE}" <<ENVEOF
export VLLM_FLASH_ATTN_SRC_DIR="${FLASH_ATTN_DIR}"
export VLLM_CUTLASS_SRC_DIR="${CUTLASS_DIR}"
export FLASH_MLA_SRC_DIR="${FLASH_MLA_DIR}"
export TRITON_KERNELS_SRC_DIR="${TRITON_DIR}/python/triton_kernels/triton_kernels"
ENVEOF

echo
echo "Pinned vLLM build dependencies are ready under: ${DEPS_DIR}"
echo "Before building vLLM, run:"
echo "  source ${ENV_FILE}"
