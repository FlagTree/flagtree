#!/bin/bash

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

TX8_HOME=$WORKSPACE/tx8_deps
LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64

if [ ! -d $TX8_HOME ] || [ ! -d $LLVM ]; then
    WORKSPACE="${HOME}/.flagtree/tsingmicro/"
    TX8_HOME=$WORKSPACE/tx8_deps
    LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64
fi

if [ ! -d $TX8_HOME ]; then
    echo "Error: $TX8_HOME not exist!" 1>&2
    exit 1
fi

if [ ! -d $LLVM ]; then
    echo "Error: $LLVM not exist!" 1>&2
    exit 1
fi

BUILD_TYPE=Release

export TX8_HOME=$TX8_HOME
export LLVM_SYSPATH=$LLVM
export FLAGTREE_BACKEND=tsingmicro

export TRITON_OFFLINE_BUILD=ON
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_WITH_CCACHE=true
export TRITON_BUILD_PROTON=OFF

echo "export TX8_HOME=$TX8_HOME"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"
echo "export FLAGTREE_BACKEND=$FLAGTREE_BACKEND"

echo "export TRITON_OFFLINE_BUILD=$TRITON_OFFLINE_BUILD"
echo "export TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD"
echo "export TRITON_BUILD_WITH_CCACHE=$TRITON_BUILD_WITH_CCACHE"
echo "export TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON"

cd python
python3 -m pip install . --no-build-isolation -v --verbose
