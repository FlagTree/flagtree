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
    ptxas_blackwell_version=$(grep '"ptxas-blackwell"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"ptxas-blackwell": "([^"]+)".*/\1/')
    ptxas_version=$(grep '"ptxas"' "$NV_TOOLCHAIN_VERSION_FILE" | grep -v "ptxas-blackwell" | sed -E 's/.*"ptxas": "([^"]+)".*/\1/')
    cuobjdump_version=$(grep '"cuobjdump"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cuobjdump": "([^"]+)".*/\1/')
    nvdisasm_version=$(grep '"nvdisasm"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"nvdisasm": "([^"]+)".*/\1/')
    cudacrt_version=$(grep '"cudacrt"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudacrt": "([^"]+)".*/\1/')
    cudart_version=$(grep '"cudart"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudart": "([^"]+)".*/\1/')
    cupti_version=$(grep '"cupti"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cupti": "([^"]+)".*/\1/')
    echo -e "Nvidia Toolchain Version Requirement:"
    echo -e "   ptxas: $ptxas_version"
    echo -e "   ptxas-blackwell: $ptxas_blackwell_version"
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
    exit 1
fi

# detect system arch
system=$(uname)
case "$system" in
    Linux)
        system="linux"
        ;;
    Darwin)
        arch="linux"
        ;;
    *)
        echo -e "${RED}Error: Unsupported current system: '$system'.${NC}"
        echo -e "   Supported system: ${GREEN}Linux, Darwin${NC}"
        exit 1
        ;;
esac
echo -e "Current System for offline building: $system"

arch=$(uname -m)
case "$arch" in
    x86_64)
        arch="x86_64"
        ;;
    arm64|aarch64)
        arch="sbsa"
        ;;
    *)
        echo -e "${RED}Error: Unsupported current system architecture '$arch'.${NC}"
        echo -e "   Supported system arch: ${GREEN}x86_64, arm64, aarch64${NC}"
        exit 1
        ;;
esac
echo -e "Current System Arch for offline building: $arch"

# handle params
if [ $# -ge 1 ]; then
    input_zip="$1"
    echo -e "${BLUE}Use $input_zip as input packed .zip file${NC}"
else
    echo -e "${RED}Error: No input .zip file specified${NC}"
    echo -e "${GREEN}Usage: sh utils/offline_build_unpack.sh [input_zip] [output_dir]${NC}"
    exit 1
fi

# handle output
if [ $# -ge 2 ]; then
    output_dir="$2"
    echo -e "${BLUE}Use $output_dir as output directory${NC}"
else
    output_dir="$HOME/.triton"
    echo -e "${YELLOW}Use default output directory: $output_dir${NC}"
    if [ -d "$output_dir" ]; then
        old_output_dir=${output_dir}.$(date +%Y%m%d_%H%M%S)
        echo -e "${YELLOW}$output_dir already exists, mv to $old_output_dir${NC}"
        mv $output_dir $old_output_dir
    fi
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

nvcc_ptxas_file="cuda-nvcc-${ptxas_version}.tar.xz"
nvcc_ptxas_blackwell_file="cuda-nvcc-${ptxas_blackwell_version}.tar.xz"
nvcc_cudacrt_file="cuda-nvcc-${cudacrt_version}.tar.xz"
cuobjdump_file="cuda-cuobjdump-${cuobjdump_version}.tar.xz"
nvdisasm_file="cuda-nvdisasm-${nvdisasm_version}.tar.xz"
cudart_file="cuda-cudart-dev-${cudart_version}.tar.xz"
cupti_file="cuda-cupti-${cupti_version}.tar.xz"
json_file="include.zip"
googletest_file="googletest-release-1.12.1.zip"
flir_file="flir-main.zip"
triton_shared_file="triton-shared-5842469a16b261e45a2c67fbfc308057622b03ee.zip"



if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi



echo -e "Unpacking ${input_zip} into ${output_dir}..."
unzip "${input_zip}" -d ${output_dir}

echo -e "Creating directory ${output_dir}/nvidia ..."
mkdir -p "${output_dir}/nvidia"

echo -e "Creating directory ${output_dir}/nvidia/nvcc ..."
mkdir -p "${output_dir}/nvidia/nvcc"
echo -e "Extracting $nvcc_ptxas_file into ${output_dir}/nvidia/nvcc ..."
tar -Jxf $output_dir/$nvcc_ptxas_file -C "${output_dir}/nvidia/nvcc"

echo -e "Extracting $nvcc_ptxas_blackwell_file into ${output_dir}/nvidia/nvcc ..."
tar -Jxf $output_dir/$nvcc_ptxas_blackwell_file -C "${output_dir}/nvidia/nvcc"

echo -e "Extracting $nvcc_cudacrt_file into ${output_dir}/nvidia/nvcc ..."
tar -Jxf $output_dir/$nvcc_cudacrt_file -C "${output_dir}/nvidia/nvcc"

echo -e "Creating directory ${output_dir}/nvidia/cuobjdump ..."
mkdir -p "${output_dir}/nvidia/cuobjdump"
echo -e "Extracting $cuobjdump_file into ${output_dir}/nvidia/cuobjdump ..."
tar -Jxf $output_dir/$cuobjdump_file -C "${output_dir}/nvidia/cuobjdump"

echo -e "Creating directory ${output_dir}/nvidia/nvdisasm ..."
mkdir -p "${output_dir}/nvidia/nvdisasm"
echo -e "Extracting $nvdisasm_file into ${output_dir}/nvidia/nvdisasm ..."
tar -Jxf $output_dir/$nvdisasm_file -C "${output_dir}/nvidia/nvdisasm"

echo -e "Creating directory ${output_dir}/nvidia/cudart ..."
mkdir -p "${output_dir}/nvidia/cudart"
echo -e "Extracting $cudart_file into ${output_dir}/nvidia/cudart ..."
tar -Jxf $output_dir/$cudart_file -C "${output_dir}/nvidia/cudart"

echo -e "Creating directory ${output_dir}/nvidia/cupti ..."
mkdir -p "${output_dir}/nvidia/cupti"
echo -e "Extracting $cupti_file into ${output_dir}/nvidia/cupti ..."
tar -Jxf $output_dir/$cupti_file -C "${output_dir}/nvidia/cupti"

echo -e "Creating directory ${output_dir}/json ..."
mkdir -p "${output_dir}/json"
echo -e "Extracting $json_file into ${output_dir}/json ..."
unzip $output_dir/$json_file -d "${output_dir}/json" > /dev/null

echo -e "Extracting $googletest_file into ${output_dir}/googletest-release-1.12.1 ..."
unzip $output_dir/$googletest_file -d "${output_dir}" > /dev/null

if [ -f "$output_dir/${flir_file}" ]; then
    echo -e "Extracting $flir_file into ${output_dir}/flir ..."
    unzip $output_dir/$flir_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/flir-main ${output_dir}/flir
else
    echo -e "${YELLOW}Warning: File $output_dir/$flir_file does not exist. This file is necessary for aipu backend, please check if you need it.${NC}"
fi

if [ -f "$output_dir/${triton_shared_file}" ]; then
    echo -e "Extracting $triton_shared_file into ${output_dir}/triton_shared ..."
    unzip $output_dir/$triton_shared_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-shared-5842469a16b261e45a2c67fbfc308057622b03ee ${output_dir}/triton_shared
else
    echo -e "${YELLOW}Warning: File $output_dir/$triton_shared_file does not exist. This file is optional, please check if you need it.${NC}"
fi

echo -e ""
echo -e "Delete $output_dir/$nvcc_ptxas_file"
rm $output_dir/$nvcc_ptxas_file
if [ -f "$output_dir/$nvcc_ptxas_file" ]; then
    echo -e "Delete $output_dir/$nvcc_ptxas_blackwell_file"
    rm $output_dir/$nvcc_ptxas_blackwell_file
fi
if [ -f "$output_dir/$nvcc_ptxas_blackwell_file" ]; then
    echo -e "Delete $output_dir/$nvcc_cudacrt_file"
    rm $output_dir/$nvcc_cudacrt_file
fi
echo -e "Delete $output_dir/$cuobjdump_file"
rm $output_dir/$cuobjdump_file
echo -e "Delete $output_dir/$nvdisasm_file"
rm $output_dir/$nvdisasm_file
echo -e "Delete $output_dir/$cudart_file"
rm $output_dir/$cudart_file
echo -e "Delete $output_dir/$cupti_file"
rm $output_dir/$cupti_file
echo -e "Delete $output_dir/$json_file"
rm $output_dir/$json_file
echo -e "Delete $output_dir/$googletest_file"
rm $output_dir/$googletest_file
if [ -f "$output_dir/${flir_file}" ]; then
    echo -e "Delete $output_dir/${flir_file}"
    rm $output_dir/${flir_file}
fi
if [ -f "$output_dir/${triton_shared_file}" ]; then
    echo -e "Delete $output_dir/$triton_shared_file"
    rm $output_dir/$triton_shared_file
fi
echo -e "Delete useless link file: ${output_dir}/nvidia/cudart/*/lib/libcudart.so"
rm ${output_dir}/nvidia/cudart/*/lib/libcudart.so

if [ -f "${output_dir}/triton_shared" ]; then
    echo -e "Delete useless link file: ${output_dir}/triton_shared/python/examples/test_annotations.py"
    rm ${output_dir}/triton_shared/python/examples/test_annotations.py

    echo -e "Delete useless link file: ${output_dir}/triton_shared/python/examples/test_core.py"
    rm ${output_dir}/triton_shared/python/examples/test_core.py
fi
