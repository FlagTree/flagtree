#!/usr/bin/env bash
#
# FlagTree riscv64 交叉编译 wheel（spacemit backend，per-Python 版本）。
#
# 用法（CI / 本地通用）:
#   bash third_party/spacemit/scripts/build_wheel_riscv64.sh \
#       <LLVM_cross> <spine-mlir-riscv-install> <spine-runtime-install> \
#       <mlir-bindings-dir> <py-riscv64-dir>
#
# 参数（全部目录，相对路径会被解析为绝对路径）:
#   1. LLVM cross install（f6ded0be == LLVM22 cross; CI 里是 llvm_installed/）
#   2. spine-mlir riscv64 install（bin/ + lib/, SPEIR 0.8.0 riscv 包）
#   3. spine-runtime install（include/ + lib/, libspert.so 来源）
#   4. MLIR Python bindings 包根（顶层含 mlir/，per-CPython 版本资产）
#   5. py-riscv64 根（include/python3.X + lib/libpython3.X.so）
#
# Overridable env vars:
#   MAX_JOBS          并行编译数（默认 2, CI 内存受限）
#   PY_RISCV_ROOT     等价于参数 5（参数优先）
#   SPINE_MLIR_BINDINGS_DIR  等价于参数 4（参数优先）
#   DIST_DIR          wheel 输出目录（默认 dist-riscv64）
#   ZLIB_LIBDIR       交叉 libz 所在目录（默认 $RISCV_ROOT_PATH/sysroot/usr/lib）
#   SKIP_PATCH        已自行应用 flagtree.patch 时设为 1（幂等逻辑会自动检测）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEMIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FT="$(cd "${SPACEMIT_DIR}/../.." && pwd)"

# --- 参数解析：相对路径在 pushd/cd 前先固化为绝对路径 ---
if [ $# -lt 5 ]; then
    echo "ERROR: need 5 args: <LLVM_cross> <spine-mlir> <spine-runtime> <mlir-bindings> <py-riscv64>" >&2
    echo "Usage: bash third_party/spacemit/scripts/build_wheel_riscv64.sh <LLVM_cross> <spine-mlir> <spine-runtime> <mlir-bindings> <py-riscv64>" >&2
    exit 1
fi
LLVM_CROSS=$(cd "${1}" && pwd)
SPINE_MLIR_INSTALL_DIR=$(cd "${2}" && pwd)
SPINE_RUNTIME_INSTALL_DIR=$(cd "${3}" && pwd)
MLIR_BINDINGS_DIR=$(cd "${4}" && pwd)
PY_RISCV_ROOT=${PY_RISCV_ROOT:-$(cd "${5}" && pwd)}

MAX_JOBS=${MAX_JOBS:-2}
DIST_DIR=${DIST_DIR:-${FT}/dist-riscv64}
TC="${SPACEMIT_DIR}/cmake/linux_riscv64.toolchain.cmake"
PATCH_FILE="${SPACEMIT_DIR}/patch/flagtree.patch"

: "${RISCV_ROOT_PATH:?RISCV_ROOT_PATH must be set (SpacemiT toolchain root)}"

echo "[spacemit] FlagTree root    : ${FT}"
echo "[spacemit] LLVM cross       : ${LLVM_CROSS}"
echo "[spacemit] spine-mlir       : ${SPINE_MLIR_INSTALL_DIR}"
echo "[spacemit] spine-runtime    : ${SPINE_RUNTIME_INSTALL_DIR}"
echo "[spacemit] mlir bindings    : ${MLIR_BINDINGS_DIR}"
echo "[spacemit] py-riscv64       : ${PY_RISCV_ROOT}"
echo "[spacemit] RISCV_ROOT_PATH  : ${RISCV_ROOT_PATH}"

# --- 前置检查（失败显式暴露） ---
for p in "${LLVM_CROSS}/bin/mlir-tblgen" \
         "${SPINE_MLIR_INSTALL_DIR}/bin/spine-opt" "${SPINE_MLIR_INSTALL_DIR}/bin/llc" \
         "${SPINE_MLIR_INSTALL_DIR}/bin/mlir-translate" "${SPINE_MLIR_INSTALL_DIR}/bin/opt" \
         "${SPINE_RUNTIME_INSTALL_DIR}/include/spert.hpp" \
         "${SPINE_RUNTIME_INSTALL_DIR}/lib/libspert.so" \
         "${MLIR_BINDINGS_DIR}/mlir" \
         "${PY_RISCV_ROOT}/include" \
         "${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++" \
         "${TC}" "${PATCH_FILE}"; do
    [ -e "$p" ] || { echo "FATAL: 缺少 $p" >&2; exit 1; }
done
echo "[spacemit] 前置检查 OK"

# --- py 版本探测（headers 目录名必须含 python3.X） ---
PY_VER=$(ls "${PY_RISCV_ROOT}/include" | grep -m1 '^python3\.')
if [ -z "${PY_VER}" ]; then
    echo "ERROR: no python3.* include dir under ${PY_RISCV_ROOT}/include" >&2
    exit 1
fi
PYLIB="${PY_RISCV_ROOT}/lib/lib${PY_VER}.so"
[ -e "${PYLIB}" ] || [ -e "${PYLIB}.1.0" ] || { echo "FATAL: 缺少 ${PYLIB}" >&2; exit 1; }
echo "[spacemit] target python    : ${PY_VER}"

# --- zlib（工具链 sysroot 默认; CI 由 .github/scripts/build_rv_zlib.sh 预装） ---
ZLIB_LIBDIR=${ZLIB_LIBDIR:-${RISCV_ROOT_PATH}/sysroot/usr/lib}
if [ ! -e "${ZLIB_LIBDIR}/libz.a" ] && [ ! -e "${ZLIB_LIBDIR}/libz.so" ]; then
    echo "ERROR: no libz under ${ZLIB_LIBDIR}" >&2
    echo "Build zlib for riscv64 into the toolchain sysroot (.github/scripts/build_rv_zlib.sh) or set ZLIB_LIBDIR" >&2
    exit 1
fi

# --- 幂等应用 flagtree.patch（对齐 install_flagtree_plugin.sh 三段逻辑） ---
cd "${FT}"
PATCH_EXCLUDES=(--exclude=python/setup_tools/utils/spacemit.py)
if git apply "${PATCH_EXCLUDES[@]}" -R --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[spacemit] flagtree.patch already applied — skipping."
elif git apply "${PATCH_EXCLUDES[@]}" --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[spacemit] applying flagtree.patch ..."
    git apply "${PATCH_EXCLUDES[@]}" "${PATCH_FILE}"
elif git apply "${PATCH_EXCLUDES[@]}" --3way "${PATCH_FILE}" 2>/tmp/spacemit_3way.log; then
    echo "[spacemit] flagtree.patch applied via 3-way merge (context had drifted)."
    if grep -Ern '^(<<<<<<< .+|=======|>>>>>>> .+)$' --include='*.py' --include='*.txt' \
         --include='*.td' --include='*.cc' --include='*.cmake' \
         CMakeLists.txt setup.py python/ include/ third_party/proton/ third_party/tle/ 2>/dev/null; then
        echo "[spacemit] ERROR: 3-way merge left conflict markers." >&2
        sed 's/^/             /' /tmp/spacemit_3way.log >&2
        exit 1
    fi
else
    echo "[spacemit] ERROR: flagtree.patch does not apply cleanly." >&2
    sed 's/^/             /' /tmp/spacemit_3way.log >&2
    exit 1
fi
# 硬校验：CMakeLists arch 表 riscv64 分支必须在位
grep -q 'CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64"' CMakeLists.txt || {
    echo "FATAL: CMakeLists.txt 缺 riscv64 分支，flagtree.patch 未生效" >&2; exit 1; }

# --- 清理上一轮 x86 build 缓存（cross 与 x86 共享 build 目录，混配必错） ---
rm -rf build/cmake.linux-x86_64-cpython-* build/lib.linux-x86_64-cpython-* build/temp.linux-x86_64-cpython-*

# --- 交叉编译（pip wheel 走 FlagTree 根 setup.py 的 spacemit 插件路径） ---
export RISCV_ROOT_PATH
export SPINE_MLIR_INSTALL_DIR
export SPINE_RUNTIME_INSTALL_DIR
# setup.py 的 bindings 探测接受「顶层含 mlir/」的单层布局（-python release 包）
export SPINE_MLIR_BINDINGS_DIR="${MLIR_BINDINGS_DIR}"
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${TC} \
-DPython3_INCLUDE_DIR:PATH=${PY_RISCV_ROOT}/include/${PY_VER} \
-DPython3_LIBRARY:FILEPATH=${PYLIB} \
-DPython3_FIND_STRATEGY=LOCATION \
-DPython3_FIND_REGISTRY=NEVER \
-DPython3_FIND_FRAMEWORK=NEVER \
-DCMAKE_SHARED_LINKER_FLAGS=-L${ZLIB_LIBDIR} \
-DCMAKE_EXE_LINKER_FLAGS=-L${ZLIB_LIBDIR}"
export CC="${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc"
export CXX="${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++"

echo "[spacemit] == 交叉构建（约 30-60 分钟）=="
FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH="${LLVM_CROSS}" \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS="${MAX_JOBS}" \
python3 -m pip wheel . --no-build-isolation -w "${DIST_DIR}"

# --- 确认编译器确实是 riscv（toolchain-file 配置无裸 CMAKE_CXX_COMPILER: 条目） ---
CMAKE_CACHE=$(ls build/cmake.linux-x86_64-cpython-*/CMakeCache.txt 2>/dev/null | head -1 || true)
if [ -z "${CMAKE_CACHE}" ]; then
    echo "[spacemit] WARNING: CMakeCache.txt not found — skipping compiler sanity check" >&2
else
    cxxline=$(grep -E '^CMAKE_CXX_COMPILER:' "${CMAKE_CACHE}" || true)
    [ -z "$cxxline" ] && cxxline=$(grep -E '^CMAKE_CXX_COMPILER_(AR|RANLIB):' "${CMAKE_CACHE}" | head -1 || true)
    echo "$cxxline"
    case "$cxxline" in
        *riscv64-unknown-linux-gnu*) echo "[spacemit] compiler OK (riscv cross)" ;;
        *) echo "[spacemit] ERROR: 编译器不是 riscv 交叉: '$cxxline'" >&2; exit 1 ;;
    esac
fi

# --- wheel 平台标签重打（pip wheel 在 x86 host 上产出 linux_x86_64 标签） ---
whl=$(ls "${DIST_DIR}"/*-linux_x86_64.whl 2>/dev/null | head -1 || true)
if [ -n "${whl}" ]; then
    python3 -m wheel tags --platform-tag linux_riscv64 --remove "${whl}" \
        || { echo "FATAL: wheel retag failed (python3 -m wheel / wheel 包可用?)" >&2; exit 1; }
fi
whl=$(ls "${DIST_DIR}"/*-linux_riscv64.whl | head -1)
echo "[spacemit] 完成: ${whl}"
