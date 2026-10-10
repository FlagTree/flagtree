# Build zlib for riscv64 and install it into the toolchain sysroot, so the
# cross-compiled triton can link -lz. Sourced/executed from the
# Spacemit3.6-Riscv64-Wheel workflow with RISCV_ROOT_PATH set.
# No set -e here on purpose: it is a plain scriptlet under the workflow step.
set -e
wget https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
tar -zxvf zlib-1.3.1.tar.gz

# Build in a subshell to avoid polluting the outer environment
(
  cd zlib-1.3.1

  # Use SpacemiT cross-compiler
  export CC=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc
  export AR=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-ar
  export RANLIB=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-ranlib

  # Install into the toolchain sysroot
  ./configure --prefix=${RISCV_ROOT_PATH}/sysroot/usr
  make -j4
  make install
)

rm -rf zlib-1.3.1.tar.gz zlib-1.3.1
