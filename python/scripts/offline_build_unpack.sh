#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

printfln() {
    printf "%b
" "$@"
}

printfln " =================== Start Unpacking Offline Build Dependencies ==================="
printfln ""
SCRIPT_DIR=$(dirname $0)
# detect pybind11 version requirement
PYBIND11_VERSION_FILE="$SCRIPT_DIR/../../cmake/pybind11-version.txt"
if [ -f "$PYBIND11_VERSION_FILE" ]; then
    pybind11_version=$(tr -d '\n' < "$PYBIND11_VERSION_FILE")
    printfln "Pybind11 Version Required: $pybind11_version"
else
    printfln "${RED}Error: version file $PYBIND11_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="$SCRIPT_DIR/../../cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    printfln "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    printfln "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
fi

# handle params
if [ $# -ge 1 ]; then
    input_zip="$1"
    printfln "${BLUE}Use $input_zip as input packed .zip file${NC}"
else
    printfln "${RED}Error: No input .zip file specified${NC}"
    printfln "${GREEN}Usage: sh utils/offline_build_unpack.sh [input_zip] [output_dir]${NC}"
    exit 1
fi

# handle output
if [ $# -ge 2 ]; then
    output_dir="$2"
    printfln "${BLUE}Use $output_dir as output directory${NC}"
else
    output_dir="$HOME/.triton"
    printfln "${YELLOW}Use default output directory: $output_dir${NC}"
    if [ -d "$output_dir" ]; then
        old_output_dir=${output_dir}.$(date +%Y%m%d_%H%M%S)
        printfln "${YELLOW}$output_dir already exists, mv to $old_output_dir${NC}"
        mv $output_dir $old_output_dir
    fi
fi

if [ ! -f "${input_zip}" ]; then
    printfln "${RED}Error: Cannot find input file $input_zip${NC}"
    exit 1
else
    printfln "Find input packed .zip file: ${input_zip}"
fi
printfln ""

if [ ! -d "$output_dir" ]; then
    printfln "Creating output directory $output_dir"
    mkdir -p "$output_dir"
else
    printfln "Output directory $output_dir already exists"
fi
printfln ""

nvcc_file="${output_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
cuobjdump_file="${output_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
nvdisam_file="${output_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
cudart_file="${output_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
cupti_file="${output_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
json_file="${output_dir}/include.zip"
pybind11_file="${output_dir}/pybind11-${pybind11_version}.tar.gz"
googletest_file="${output_dir}/googletest-release-1.12.1.zip"
triton_shared_file="${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"



if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
fi



printfln "Unpacking ${input_zip} into ${output_dir}..."
unzip "${input_zip}" -d ${output_dir}

printfln "Creating directory ${output_dir}/nvidia ..."
mkdir "${output_dir}/nvidia"

printfln "Creating directory ${output_dir}/nvidia/ptxas ..."
mkdir "${output_dir}/nvidia/ptxas"
printfln "Extracting $nvcc_file into ${output_dir}/nvidia/ptxas ..."
tar -jxf $nvcc_file -C "${output_dir}/nvidia/ptxas"

printfln "Creating directory ${output_dir}/nvidia/cuobjdump ..."
mkdir "${output_dir}/nvidia/cuobjdump"
printfln "Extracting $cuobjdump_file into ${output_dir}/nvidia/cuobjdump ..."
tar -jxf $cuobjdump_file -C "${output_dir}/nvidia/cuobjdump"

printfln "Creating directory ${output_dir}/nvidia/nvdisasm ..."
mkdir "${output_dir}/nvidia/nvdisasm"
printfln "Extracting $nvdisam_file into ${output_dir}/nvidia/nvdisasm ..."
tar -jxf $nvdisam_file -C "${output_dir}/nvidia/nvdisasm"

printfln "Creating directory ${output_dir}/nvidia/cudacrt ..."
mkdir "${output_dir}/nvidia/cudacrt"
printfln "Extracting $nvcc_file into ${output_dir}/nvidia/cudacrt ..."
tar -jxf $nvcc_file -C "${output_dir}/nvidia/cudacrt"

printfln "Creating directory ${output_dir}/nvidia/cudart ..."
mkdir "${output_dir}/nvidia/cudart"
printfln "Extracting $cudart_file into ${output_dir}/nvidia/cudart ..."
tar -jxf $cudart_file -C "${output_dir}/nvidia/cudart"

printfln "Creating directory ${output_dir}/nvidia/cupti ..."
mkdir "${output_dir}/nvidia/cupti"
printfln "Extracting $cupti_file into ${output_dir}/nvidia/cupti ..."
tar -jxf $cupti_file -C "${output_dir}/nvidia/cupti"

printfln "Creating directory ${output_dir}/json ..."
mkdir "${output_dir}/json"
printfln "Extracting $json_file into ${output_dir}/json ..."
unzip $json_file -d "${output_dir}/json" > /dev/null

printfln "Creating directory ${output_dir}/pybind11 ..."
mkdir "${output_dir}/pybind11"
printfln "Extracting $pybind11_file into ${output_dir}/pybind11 ..."
tar -zxf $pybind11_file -C "${output_dir}/pybind11"

printfln "Extracting $googletest_file into ${output_dir}/googletest-release-1.12.1 ..."
unzip $googletest_file -d "${output_dir}" > /dev/null

if [ -f "${triton_shared_file}" ]; then
    printfln "Extracting $triton_shared_file into ${output_dir}/triton_shared ..."
    unzip $triton_shared_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33 ${output_dir}/triton_shared
else
    printfln "Warning: File $triton_shared_file does not exist. This file is optional, please check if you need it."
fi

printfln ""
printfln "Delete $nvcc_file"
rm $nvcc_file
printfln "Delete $cuobjdump_file"
rm $cuobjdump_file
printfln "Delete $nvdisam_file"
rm $nvdisam_file
printfln "Delete $cudart_file"
rm $cudart_file
printfln "Delete $cupti_file"
rm $cupti_file
printfln "Delete $json_file"
rm $json_file
printfln "Delete $pybind11_file"
rm $pybind11_file
printfln "Delete $googletest_file"
rm $googletest_file
if [ -f "${triton_shared_file}" ]; then
    printfln "Delete $triton_shared_file"
    rm $triton_shared_file
fi
printfln "Delete useless file: ${output_dir}/nvidia/cudart/lib/libcudart.so"
rm ${output_dir}/nvidia/cudart/lib/libcudart.so
