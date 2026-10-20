#!/usr/bin/env bash
# Build library=cannshmem_mlir device bitcode from AscendNPU-IR Shmem template.
# Requires: source CANN set_env.sh (ccec / llvm-link on PATH).
#
# Output: lib/libdevice_cannshmem_mlir*.bc  (aiv + mix_aiv + mix_aic, noinline wrappers)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/lib"
mkdir -p "${OUT_DIR}"

# Prefer the shmem-enabled AscendNPU-IR tree used for template builds.
ROOT_FLASH="$(cd -- "${SCRIPT_DIR}/../../../../../../../.." && pwd)"
SHMEM_IR="${FLAGTREE_ASCEND_SHMEM_IR:-${ROOT_FLASH}/FlagTree-AscendNPU-IR-shmem}"
if [[ ! -f "${SHMEM_IR}/bishengir/lib/Template/lib/Shmem/Shmem.cpp" ]]; then
  SHMEM_IR="$(cd -- "${SCRIPT_DIR}/../../../../../../../../third_party/ascend/AscendNPU-IR" && pwd)"
fi
TEMPLATE_INCLUDE="${SHMEM_IR}/bishengir/lib/Template/include"
SHMEM_CPP="${SHMEM_IR}/bishengir/lib/Template/lib/Shmem/Shmem.cpp"
SHMEM_INC="${SHMEM_IR}/third-party/shmem/include"
SHMEM_SRC_DEV="${SHMEM_IR}/third-party/shmem/src/device"

ASCEND="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
CCEC="${CCEC:-$(command -v ccec || true)}"
CCEC="${CCEC:-${ASCEND}/bin/ccec}"
LINK="${LLVM_LINK:-$(dirname -- "${CCEC}")/llvm-link}"
if [[ ! -x "${LINK}" ]]; then
  LINK="${ASCEND}/tools/bisheng_compiler/bin/llvm-link"
fi

for p in "${SHMEM_CPP}" "${TEMPLATE_INCLUDE}" "${SHMEM_INC}" "${CCEC}" "${LINK}"; do
  if [[ ! -e "${p}" ]]; then
    echo "error: missing ${p}" >&2
    exit 1
  fi
done

# Strip always_inline on mlir ciface wrappers so symbols survive late linking.
NOINLINE_CPP="${OUT_DIR}/Shmem.noinline.cpp"
python3 - <<PY
from pathlib import Path
src = Path("${SHMEM_CPP}").read_text()
dst = src.replace("__attribute__((always_inline)) ", "").replace("__attribute__((always_inline))", "")
Path("${NOINLINE_CPP}").write_text(dst)
print("wrote", "${NOINLINE_CPP}", "removed", src.count("always_inline") - dst.count("always_inline"), "always_inline")
PY

COMMON=(
  -O2 -x cce -emit-llvm -c
  --cce-auto-sync=off --cce-aicore-only --cce-generic-addrspace=off
  -mllvm -disable-llvm-optzns --cce-enable-print --cce-enable-sanitizer
  -std=c++17
  -I "${TEMPLATE_INCLUDE}"
  -I "${SHMEM_INC}"
  -I "${SHMEM_SRC_DEV}"
  -I "${ASCEND}/aarch64-linux/include"
  -I "${ASCEND}/compiler/tikcpp/tikcfw"
  -I "${ASCEND}/compiler/tikcpp/tikcfw/impl"
  -I "${ASCEND}/compiler/tikcpp/tikcfw/interface"
)

compile_one() {
  local arch="$1"
  local out="$2"
  shift 2
  echo "== ccec ${arch} -> $(basename "${out}")"
  "${CCEC}" "${COMMON[@]}" --cce-aicore-arch="${arch}" "$@" \
    "${NOINLINE_CPP}" -o "${out}"
}

# Keep each ccec variant as its own .bc. Shared globals (e.g. g_tilingKey)
# collide under llvm-link; Ascend's --link-aicore-bitcode accepts multiple
# paths and picks the needed .vector / .cube symbols per kernel.
compile_one dav-c220-vec "${OUT_DIR}/libdevice_cannshmem_mlir.aiv.bc"
compile_one dav-c220-vec "${OUT_DIR}/libdevice_cannshmem_mlir.mix.aiv.bc" \
  -cce-enable-mix -mllvm -enable-mix=true
compile_one dav-c220-cube "${OUT_DIR}/libdevice_cannshmem_mlir.mix.aic.bc" \
  -cce-enable-mix -mllvm -enable-mix=true

# Default single-file alias for FLAGTREE_CANNSHMEM_MLIR_DEVICE_BC (vector path).
cp -f "${OUT_DIR}/libdevice_cannshmem_mlir.aiv.bc" "${OUT_DIR}/libdevice_cannshmem_mlir.bc"

ls -lh \
  "${OUT_DIR}/libdevice_cannshmem_mlir.aiv.bc" \
  "${OUT_DIR}/libdevice_cannshmem_mlir.mix.aiv.bc" \
  "${OUT_DIR}/libdevice_cannshmem_mlir.mix.aic.bc"
echo "OK"
