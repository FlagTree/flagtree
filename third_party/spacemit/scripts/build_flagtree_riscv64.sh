#!/usr/bin/env bash
#
# FlagTree riscv64 交叉编译 wheel（spacemit backend，2026-09-29）
#
# What it does:
#   0. 前置检查：所有依赖路径必须存在
#   1. 固化 riscv spine-mlir 工具进 .riscv-assets/（防 ci-builds 轮转清理）
#   2. 组装 riscv spine-runtime 0.6.3（libspert + include）
#   3. 备份并清理 x86 构建缓存/backend 二进制（硬校验 riscv64 CMakeLists 分支在位）
#   4. pip wheel 交叉编译（约 30-60 分钟）
#   5. 校验 CMAKE_CXX_COMPILER 确为 riscv 交叉编译器
#   6. wheel 平台标签重打为 linux_riscv64 + 内容验证
#   7. 恢复源码树为 x86 状态
#
# Usage (from anywhere):
#   bash third_party/spacemit/scripts/build_flagtree_riscv64.sh 2>&1 | tee ~/tmp/ft_riscv_build.log
#
# Overridable env vars（默认 = 本机 NFS / CI runner workspace 布局）:
#   W                  CI runner workspace（.venv-ci / llvm_installed / rpc_runtime_installed /
#                      spine_mlir_installed 所在），默认 <FlagTree 上级目录>/github-ci/actions-runner/_work/FlagTree/FlagTree
#   LLVM_CROSS         riscv64 交叉编译 LLVM（f6ded0be）
#   SML_SRC            riscv spine-mlir installed 源（ci-builds 产物，步骤 1 固化进 .riscv-assets）
#   SPINE_RUNTIME_SRC  spine-runtime 源码仓（取 include/）
#   TOOLCHAIN_ROOT     SpacemiT 交叉工具链（导出为 RISCV_ROOT_PATH）
#   PYINC              riscv Python 头文件目录（交叉编译 Python.h）
#   ZLIB_LIBDIR        静态交叉编译 libz.a 所在目录
#   MAX_JOBS           编译并发数（默认 64）
#   TMPDIR             构建临时目录（默认 $HOME/tmp）
#
set -euo pipefail

# --- 自定位 FlagTree 根（本脚本位于 third_party/spacemit/scripts/） ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEMIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FT="$(cd "${SPACEMIT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${FT}/.." && pwd)"

# --- 外部依赖（全部可用 env 覆盖） ---
W="${W:-$WORKSPACE/github-ci/actions-runner/_work/FlagTree/FlagTree}"
LLVM_CROSS="${LLVM_CROSS:-/share/nfs_share/llvm-pre-build/llvm-f6ded0be897e2878612dd903f7e8bb85448269e5-20260928-release-cross}"
SML_SRC="${SML_SRC:-/share/nfs_share/ci-builds/ai/AICompiler/spine-mlir/24462/build-riscv64/installed}"
SPINE_RUNTIME_SRC="${SPINE_RUNTIME_SRC:-$WORKSPACE/spine-runtime}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/share/nfs_share/ci-node-compiler-env/spacemit-toolchain-linux-glibc-x86_64-v1.1.2}"
PYINC="${PYINC:-/share/nfs_share/env/python3.12.7/include/python3.12}"
ZLIB_LIBDIR="${ZLIB_LIBDIR:-$HOME/work/riscv-deps/zlib/lib}"
MAX_JOBS="${MAX_JOBS:-64}"
TMPDIR="${TMPDIR:-$HOME/tmp}"

# --- 派生路径 ---
SML="$FT/.riscv-assets/spine_mlir_installed"
RT="$FT/.riscv-assets/spine_runtime_installed"
TC="$SPACEMIT_DIR/cmake/linux_riscv64.toolchain.cmake"

echo "[spacemit] FlagTree root  : $FT"
echo "[spacemit] CI workspace   : $W"
echo "[spacemit] LLVM cross     : $LLVM_CROSS"
echo "[spacemit] spine-mlir src : $SML_SRC"
echo "[spacemit] toolchain      : $TOOLCHAIN_ROOT"

echo "[spacemit] == [0/7] 前置检查：所有依赖路径必须存在 =="
for p in "$LLVM_CROSS/bin/mlir-tblgen" "$SML_SRC/bin/spine-opt" "$SML_SRC/bin/llc" \
         "$SML_SRC/bin/mlir-translate" "$SML_SRC/bin/opt" "$TC" \
         "$W/rpc_runtime_installed/lib/libspert.so.0.6.3" \
         "$SPINE_RUNTIME_SRC/include/spert.hpp" \
         "$TOOLCHAIN_ROOT/bin/riscv64-unknown-linux-gnu-g++" "$PYINC/Python.h" \
         "$W/.venv-ci/bin/activate" "$W/llvm_installed/bin" \
         "$W/spine_mlir_installed/bin/spine-opt" "$ZLIB_LIBDIR"; do
  [ -e "$p" ] || { echo "FATAL: 缺少 $p" >&2; exit 1; }
done
echo "[spacemit] 前置检查 OK"

echo "[spacemit] == [1/7] 固化 riscv spine-mlir 工具（防 ci-builds 轮转清理）=="
mkdir -p "$FT/.riscv-assets"
[ -d "$SML/bin" ] || cp -a "$SML_SRC" "$SML"

echo "[spacemit] == [2/7] 组装 riscv spine-runtime 0.6.3 =="
mkdir -p "$RT/lib"
cp -a "$W/rpc_runtime_installed/lib/libspert.so"* "$RT/lib/"
[ -d "$RT/include" ] || cp -a "$SPINE_RUNTIME_SRC/include" "$RT/"

echo "[spacemit] == [3/7] 备份并清理 x86 构建缓存/backend 二进制 =="
cd "$FT"
# 硬校验：riscv cross 构建关键依赖的 CMakeLists arch 表 riscv64 分支必须在位
#（旧 flagtree.patch 时代的 hunk 现已直接提交进根 CMakeLists.txt）
grep -q 'CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64"' CMakeLists.txt || {
  echo "FATAL: CMakeLists.txt 缺 riscv64 分支" >&2; exit 1; }
echo "[spacemit] riscv64 CMakeLists 分支在位"

mkdir -p .riscv-assets/backup-backend
[ -d .riscv-assets/backup-backend/bin ] || cp -a third_party/spacemit/backend/bin .riscv-assets/backup-backend/bin
[ -d .riscv-assets/backup-backend/lib ] || cp -a third_party/spacemit/backend/lib .riscv-assets/backup-backend/lib
rm -rf build/cmake.linux-x86_64-cpython-3.12 build/lib.linux-x86_64-cpython-312 build/temp.linux-x86_64-cpython-312
rm -f third_party/spacemit/backend/bin/llc third_party/spacemit/backend/bin/mlir-translate \
      third_party/spacemit/backend/bin/opt third_party/spacemit/backend/bin/spine-opt \
      third_party/spacemit/backend/bin/spine-triton-opt
rm -f third_party/spacemit/backend/lib/libspert* third_party/spacemit/backend/lib/libSpeIR*

echo "[spacemit] == [4/7] 交叉构建（约 30-60 分钟）=="
echo "    （启动 1 分钟后可在另一终端哨兵检查：grep CMAKE_CXX_COMPILER_ $FT/build/cmake.linux-x86_64-cpython-3.12/CMakeCache.txt"
echo "      路径必须含 riscv64-unknown-linux-gnu，若是 /usr/bin 立即 Ctrl-C）"
mkdir -p "$TMPDIR"
source "$W/.venv-ci/bin/activate"
export RISCV_ROOT_PATH="$TOOLCHAIN_ROOT"
export TMPDIR
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$TC -DPython3_INCLUDE_DIR=$PYINC -DCMAKE_SHARED_LINKER_FLAGS=-L$ZLIB_LIBDIR -DCMAKE_EXE_LINKER_FLAGS=-L$ZLIB_LIBDIR"
echo "[spacemit] TRITON_APPEND_CMAKE_ARGS=$TRITON_APPEND_CMAKE_ARGS"

FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH="$LLVM_CROSS" \
SPINE_MLIR_INSTALL_DIR="$SML" \
SPINE_RUNTIME_INSTALL_DIR="$RT" \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS="$MAX_JOBS" \
pip wheel . --no-build-isolation -w dist-riscv64

echo "[spacemit] == [5/7] 确认编译器确实是 riscv =="
# 注意：toolchain-file 配置的 CMakeCache 没有裸 CMAKE_CXX_COMPILER: 条目
# （只有 CMAKE_CXX_COMPILER_AR/RANLIB），直接 grep 会 miss 并在 set -e 下静默退出。
cxxline=$(grep -E '^CMAKE_CXX_COMPILER:' build/cmake.linux-x86_64-cpython-3.12/CMakeCache.txt || true)
[ -z "$cxxline" ] && cxxline=$(grep -E '^CMAKE_CXX_COMPILER_(AR|RANLIB):' build/cmake.linux-x86_64-cpython-3.12/CMakeCache.txt | head -1 || true)
echo "$cxxline"
case "$cxxline" in
  *riscv64-unknown-linux-gnu*) echo "[spacemit] compiler OK (riscv cross)" ;;
  *) echo "[spacemit] ERROR: 编译器不是 riscv 交叉: '$cxxline'" >&2; exit 1 ;;
esac

echo "[spacemit] == [6/7] 重打平台标签 + 验证 wheel 内容 =="
whl=$(ls dist-riscv64/*-linux_x86_64.whl)
python3 -m wheel tags --platform-tag linux_riscv64 --remove "$whl"
whl=$(ls dist-riscv64/*-linux_riscv64.whl)
chk="$TMPDIR/whlchk"
rm -rf "$chk"; mkdir -p "$chk"
unzip -o -q "$FT/$whl" -d "$chk" 'triton/_C/libtriton*' 'triton/backends/spacemit/bin/*' 'triton/backends/spacemit/lib/*'
so=$(find "$chk" -name 'libtriton*.so' | head -1)
echo "-- libtriton.so: $(readelf -h "$so" | grep Machine)"
echo "-- backend/bin:"; file "$chk"/triton/backends/spacemit/bin/* | grep -v __init__ || true
echo "-- backend/lib:"; ls -la "$chk/triton/backends/spacemit/lib/"

echo "[spacemit] == [7/7] 恢复源码树为 x86 状态 =="
rm -rf third_party/spacemit/backend/bin third_party/spacemit/backend/lib
cp -a .riscv-assets/backup-backend/bin third_party/spacemit/backend/bin
cp -a .riscv-assets/backup-backend/lib third_party/spacemit/backend/lib
cp "$W/spine_mlir_installed/bin/spine-opt" third_party/spacemit/backend/bin/spine-opt
ln -sfn "$W/llvm_installed" "$HOME/.triton/llvm/llvm-ubuntu-x64"
echo "[spacemit] 完成: $FT/$whl"
