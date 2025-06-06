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

export TX8_HOME=$TX8_HOME
export LLVM_SYSPATH=$LLVM
export LD_LIBRARY_PATH=$TX8_HOME/lib:$LD_LIBRARY_PATH
export TRITON_ALWAYS_COMPILE=1

# export TRITON_DUMP_PATH=$project_dir/dump

echo "export TX8_HOME=$TX8_HOME"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "export TRITON_ALWAYS_COMPILE=$TRITON_ALWAYS_COMPILE"

python3 $@
