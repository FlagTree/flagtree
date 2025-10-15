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
# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="$SCRIPT_DIR/../../cmake/nvidia-toolchain-version.json"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    ptxas_version=$(grep '"ptxas"' "$NV_TOOLCHAIN_VERSION_FILE" | grep -v "ptxas-blackwell" | sed -E 's/.*"ptxas": "([^"]+)".*/\1/')
    cuobjdump_version=$(grep '"cuobjdump"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cuobjdump": "([^"]+)".*/\1/')
    nvdisasm_version=$(grep '"nvdisasm"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"nvdisasm": "([^"]+)".*/\1/')
    cudacrt_version=$(grep '"cudacrt"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudacrt": "([^"]+)".*/\1/')
    cudart_version=$(grep '"cudart"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudart": "([^"]+)".*/\1/')
    cupti_version=$(grep '"cupti"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cupti": "([^"]+)".*/\1/')
    printfln "Nvidia Toolchain Version Requirement:"
    printfln "   ptxas: $ptxas_version"
    printfln "   cuobjdump: $cuobjdump_version"
    printfln "   nvdisasm: $nvdisasm_version"
    printfln "   cudacrt: $cudacrt_version"
    printfln "   cudart: $cudart_version"
    printfln "   cupti: $cupti_version"
else
    printfln "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect json version requirement
JSON_VERSION_FILE="$SCRIPT_DIR/../../cmake/json-version.txt"
if [ -f "$JSON_VERSION_FILE" ]; then
    json_version=$(tr -d '\n' < "$JSON_VERSION_FILE")
    printfln "JSON Version Required: $json_version"
else
    printfln "${RED}Error: version file $JSON_VERSION_FILE is not exist${NC}"
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

ptxas_file=${output_dir}/cuda-ptxas-${ptxas_version}-0.tar.bz2
cudacrt_file=${output_dir}/cuda-crt-${cudacrt_version}-0.tar.bz2
cuobjdump_file="${output_dir}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2"
nvdisasm_file="${output_dir}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2"
cudart_file="${output_dir}/cuda-cudart-dev-${cudart_version}-0.tar.bz2"
cupti_file="${output_dir}/cuda-cupti-${cupti_version}-0.tar.bz2"
json_file="${output_dir}/include.zip"
googletest_file="${output_dir}/googletest-release-1.12.1.zip"
trtion_ascend_file="${output_dir}/triton-ascend-master.zip"
ascendnpu_ir_file="${output_dir}/ascendnpu-ir-1922371c42749fda534d6395b7ed828b5c9f36d4.zip"
triton_file="${output_dir}/triton-9641643da6c52000c807b5eeed05edaec4402a67.zip"
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
printfln "Extracting $ptxas_file into ${output_dir}/nvidia/ptxas ..."
tar -jxf $ptxas_file -C "${output_dir}/nvidia/ptxas"

printfln "Creating directory ${output_dir}/nvidia/cudacrt ..."
mkdir "${output_dir}/nvidia/cudacrt"
printfln "Extracting $cudacrt_file into ${output_dir}/nvidia/cudacrt ..."
tar -jxf $cudacrt_file -C "${output_dir}/nvidia/cudacrt"

printfln "Creating directory ${output_dir}/nvidia/cuobjdump ..."
mkdir "${output_dir}/nvidia/cuobjdump"
printfln "Extracting $cuobjdump_file into ${output_dir}/nvidia/cuobjdump ..."
tar -jxf $cuobjdump_file -C "${output_dir}/nvidia/cuobjdump"

printfln "Creating directory ${output_dir}/nvidia/nvdisasm ..."
mkdir "${output_dir}/nvidia/nvdisasm"
printfln "Extracting $nvdisasm_file into ${output_dir}/nvidia/nvdisasm ..."
tar -jxf $nvdisasm_file -C "${output_dir}/nvidia/nvdisasm"

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

printfln "Extracting $googletest_file into ${output_dir}/googletest-release-1.12.1 ..."
unzip $googletest_file -d "${output_dir}" > /dev/null

if [ -f "${trtion_ascend_file}" ]; then
    printfln "Extracting $trtion_ascend_file into ${output_dir}/triton-ascend-master ..."
    unzip $trtion_ascend_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-ascend-master ${output_dir}/ascend

    if [ -f "${ascendnpu_ir_file}" ]; then
        printfln "Extracting $ascendnpu_ir_file into ${output_dir}/ascend/third_party/ ..."
        unzip $ascendnpu_ir_file -d "${output_dir}" > /dev/null
        mv "${output_dir}/ascendnpu-ir-1922371c42749fda534d6395b7ed828b5c9f36d4" "${output_dir}/ascendnpu-ir"
    else
        printfln "Warning: File $ascendnpu_ir_file does not exist. This file is necessary for ascend backend, please check if you need it."
    fi

    if [ -f "${triton_file}" ]; then
        printfln "Extracting $triton_file into ${output_dir}/ascend/third_party/ ..."
        unzip $triton_file -d "${output_dir}/ascend/third_party/" > /dev/null
        rm -rf "${output_dir}/ascend/third_party/triton"
        mv "${output_dir}/ascend/third_party/triton-9641643da6c52000c807b5eeed05edaec4402a67" "${output_dir}/ascend/third_party/triton"
    else
        printfln "Warning: File $ascendnpu_ir_file does not exist. This file is necessary for ascend backend, please check if you need it."
    fi

else
    printfln "Warning: File $trtion_ascend_file does not exist. This file is necessary for ascend backend, please check if you need it."
fi

if [ -f "${triton_shared_file}" ]; then
    printfln "Extracting $triton_shared_file into ${output_dir}/triton_shared ..."
    unzip $triton_shared_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33 ${output_dir}/triton_shared
else
    printfln "Warning: File $triton_shared_file does not exist. This file is optional, please check if you need it."
fi

printfln ""
printfln "Delete $ptxas_file"
rm $ptxas_file
if [ -f "${cudacrt_file}" ]; then
    printfln "Delete $cudacrt_file"
    rm $cudacrt_file
fi
printfln "Delete $cuobjdump_file"
rm $cuobjdump_file
printfln "Delete $nvdisasm_file"
rm $nvdisasm_file
printfln "Delete $cudart_file"
rm $cudart_file
printfln "Delete $cupti_file"
rm $cupti_file
printfln "Delete $json_file"
rm $json_file
printfln "Delete $googletest_file"
rm $googletest_file
if [ -f "${trtion_ascend_file}" ]; then
    printfln "Delete $trtion_ascend_file"
    rm $trtion_ascend_file
    printfln "Delete $ascendnpu_ir_file"
    rm $ascendnpu_ir_file
    printfln "Delete $triton_file"
    rm $triton_file
fi
if [ -f "${triton_shared_file}" ]; then
    printfln "Delete $triton_shared_file"
    rm $triton_shared_file
fi
printfln "Delete useless file: ${output_dir}/nvidia/cudart/lib/libcudart.so"
rm ${output_dir}/nvidia/cudart/lib/libcudart.so
