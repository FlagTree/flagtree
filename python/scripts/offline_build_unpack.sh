#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Start Unpacking Offline Build Dependencies ==================="
echo -e ""

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="../cmake/nvidia-toolchain-version.json"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    ptxas_version=$(grep '"ptxas"' "$NV_TOOLCHAIN_VERSION_FILE" | grep -v "ptxas-blackwell" | sed -E 's/.*"ptxas": "([^"]+)".*/\1/')
    cuobjdump_version=$(grep '"cuobjdump"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cuobjdump": "([^"]+)".*/\1/')
    nvdisasm_version=$(grep '"nvdisasm"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"nvdisasm": "([^"]+)".*/\1/')
    cudacrt_version=$(grep '"cudacrt"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudacrt": "([^"]+)".*/\1/')
    cudart_version=$(grep '"cudart"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudart": "([^"]+)".*/\1/')
    cupti_version=$(grep '"cupti"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cupti": "([^"]+)".*/\1/')
    echo -e "Nvidia Toolchain Version Requirement:"
    echo -e "   ptxas: $ptxas_version"
    echo -e "   cuobjdump: $cuobjdump_version"
    echo -e "   nvdisasm: $nvdisasm_version"
    echo -e "   cudacrt: $cudacrt_version"
    echo -e "   cudart: $cudart_version"
    echo -e "   cupti: $cupti_version"
else
    echo -e "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect json version requirement
JSON_VERSION_FILE="../cmake/json-version.txt"
if [ -f "$JSON_VERSION_FILE" ]; then
    json_version=$(tr -d '\n' < "$JSON_VERSION_FILE")
    echo -e "JSON Version Required: $json_version"
else
    echo -e "${RED}Error: version file $JSON_VERSION_FILE is not exist${NC}"
fi

# handle params
if [ $# -ge 2 ]; then
    input_zip="$1"
    output_dir="$2"
    echo -e "${BLUE}Use $input_zip as input packed .zip file${NC}"
    echo -e "${BLUE}Use $output_dir as output directory${NC}"
else
    echo -e "${RED}Error: No input file or output directory specified${NC}"
    echo -e "${GREEN}Usage: sh utils/offline_build_unpack.sh [input_zip] [output_dir]${NC}"
    exit 1
fi

if [ ! -f "${input_zip}" ]; then
    echo -e "${RED}Error: Cannot find input file $input_zip${NC}"
    exit 1
else
    echo -e "Find input packed .zip file: ${input_zip}"
fi
echo -e ""

if [ ! -d "$output_dir" ]; then
    echo -e "Creating output directory $output_dir"
    mkdir -p "$output_dir"
else
    echo -e "Output directory $output_dir already exists"
fi
echo -e ""

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



echo -e "Unpacking ${input_zip} into ${output_dir}..."
unzip "${input_zip}" -d ${output_dir}

echo -e "Creating directory ${output_dir}/nvidia ..."
mkdir "${output_dir}/nvidia"

echo -e "Creating directory ${output_dir}/nvidia/ptxas ..."
mkdir "${output_dir}/nvidia/ptxas"
echo -e "Extracting $ptxas_file into ${output_dir}/nvidia/ptxas ..."
tar -jxf $ptxas_file -C "${output_dir}/nvidia/ptxas"

echo -e "Creating directory ${output_dir}/nvidia/cudacrt ..."
mkdir "${output_dir}/nvidia/cudacrt"
echo -e "Extracting $cudacrt_file into ${output_dir}/nvidia/cudacrt ..."
tar -jxf $cudacrt_file -C "${output_dir}/nvidia/cudacrt"

echo -e "Creating directory ${output_dir}/nvidia/cuobjdump ..."
mkdir "${output_dir}/nvidia/cuobjdump"
echo -e "Extracting $cuobjdump_file into ${output_dir}/nvidia/cuobjdump ..."
tar -jxf $cuobjdump_file -C "${output_dir}/nvidia/cuobjdump"

echo -e "Creating directory ${output_dir}/nvidia/nvdisasm ..."
mkdir "${output_dir}/nvidia/nvdisasm"
echo -e "Extracting $nvdisasm_file into ${output_dir}/nvidia/nvdisasm ..."
tar -jxf $nvdisasm_file -C "${output_dir}/nvidia/nvdisasm"

echo -e "Creating directory ${output_dir}/nvidia/cudart ..."
mkdir "${output_dir}/nvidia/cudart"
echo -e "Extracting $cudart_file into ${output_dir}/nvidia/cudart ..."
tar -jxf $cudart_file -C "${output_dir}/nvidia/cudart"

echo -e "Creating directory ${output_dir}/nvidia/cupti ..."
mkdir "${output_dir}/nvidia/cupti"
echo -e "Extracting $cupti_file into ${output_dir}/nvidia/cupti ..."
tar -jxf $cupti_file -C "${output_dir}/nvidia/cupti"

echo -e "Creating directory ${output_dir}/json ..."
mkdir "${output_dir}/json"
echo -e "Extracting $json_file into ${output_dir}/json ..."
unzip $json_file -d "${output_dir}/json" > /dev/null

echo -e "Extracting $googletest_file into ${output_dir}/googletest-release-1.12.1 ..."
unzip $googletest_file -d "${output_dir}" > /dev/null

if [ -f "${trtion_ascend_file}" ]; then
    echo -e "Extracting $trtion_ascend_file into ${output_dir}/triton-ascend-master ..."
    unzip $trtion_ascend_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-ascend-master ${output_dir}/ascend

    if [ -f "${ascendnpu_ir_file}" ]; then
        echo -e "Extracting $ascendnpu_ir_file into ${output_dir}/ascend/third_party/ ..."
        unzip $ascendnpu_ir_file -d "${output_dir}" > /dev/null
        mv "${output_dir}/ascendnpu-ir-1922371c42749fda534d6395b7ed828b5c9f36d4" "${output_dir}/ascendnpu-ir"
    else
        echo -e "Warning: File $ascendnpu_ir_file does not exist. This file is necessary for ascend backend, please check if you need it."
    fi

    if [ -f "${triton_file}" ]; then
        echo -e "Extracting $triton_file into ${output_dir}/ascend/third_party/ ..."
        unzip $triton_file -d "${output_dir}/ascend/third_party/" > /dev/null
        rm -rf "${output_dir}/ascend/third_party/triton"
        mv "${output_dir}/ascend/third_party/triton-9641643da6c52000c807b5eeed05edaec4402a67" "${output_dir}/ascend/third_party/triton"
    else
        echo -e "Warning: File $ascendnpu_ir_file does not exist. This file is necessary for ascend backend, please check if you need it."
    fi

else
    echo -e "Warning: File $trtion_ascend_file does not exist. This file is necessary for ascend backend, please check if you need it."
fi

if [ -f "${triton_shared_file}" ]; then
    echo -e "Extracting $triton_shared_file into ${output_dir}/triton_shared ..."
    unzip $triton_shared_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33 ${output_dir}/triton_shared
else
    echo -e "Warning: File $triton_shared_file does not exist. This file is optional, please check if you need it."
fi

echo -e ""
echo -e "Delete $ptxas_file"
rm $ptxas_file
if [ -f "${cudacrt_file}" ]; then
    echo -e "Delete $cudacrt_file"
    rm $cudacrt_file
fi
echo -e "Delete $cuobjdump_file"
rm $cuobjdump_file
echo -e "Delete $nvdisasm_file"
rm $nvdisasm_file
echo -e "Delete $cudart_file"
rm $cudart_file
echo -e "Delete $cupti_file"
rm $cupti_file
echo -e "Delete $json_file"
rm $json_file
echo -e "Delete $googletest_file"
rm $googletest_file
if [ -f "${trtion_ascend_file}" ]; then
    echo -e "Delete $trtion_ascend_file"
    rm $trtion_ascend_file
    echo -e "Delete $ascendnpu_ir_file"
    rm $ascendnpu_ir_file
    echo -e "Delete $triton_file"
    rm $triton_file
fi
if [ -f "${triton_shared_file}" ]; then
    echo -e "Delete $triton_shared_file"
    rm $triton_shared_file
fi
echo -e "Delete useless file: ${output_dir}/nvidia/cudart/lib/libcudart.so"
rm ${output_dir}/nvidia/cudart/lib/libcudart.so
