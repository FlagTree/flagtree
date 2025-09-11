#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Start Packing Downloaded Offline Build Files ==================="
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

output_zip="offline-build-pack-triton-3.3.x.zip"

# handle input
echo -e ""
if [ $# -ge 1 ]; then
    input_dir="$1"
    echo -e "${BLUE}Use $input_dir as input directory${NC}"
else
    echo -e "${RED}Error: No input directory specified${NC}"
    echo -e "${GREEN}Usage: sh utils/offline_build_pack.sh [input_dir] [output_zip_file]${NC}"
    exit 1
fi

# handle output
if [ $# -ge 2 ]; then
    output_zip="$2"
    echo -e "${BLUE}Use $output_zip as output .zip file${NC}"
else
    echo -e "${YELLOW}Use default output .zip file name: $output_zip${NC}"
fi

if [ ! -d "$input_dir" ]; then
    echo -e "${RED}Error: Cannot find input directory $input_dir${NC}"
    exit 1
else
    echo -e "Find input directory: $input_dir"
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
triton_shared_file="triton-shared-5842469a16b261e45a2c67fbfc308057622b03ee.zip"

if [ ! -d "$input_dir" ]; then
    echo -e "${RED}Error: offline build download directory $input_dir does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find ${nvcc_file}"

if [ ! -f "$input_dir/$cuobjdump_file" ]; then
    echo -e "${RED}Error: File $input_dir/$cuobjdump_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$cuobjdump_file"

if [ ! -f "$input_dir/$nvdisasm_file" ]; then
    echo -e "${RED}Error: File $input_dir/$nvdisasm_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$nvdisasm_file"

if [ ! -f "$input_dir/$cudart_file" ]; then
    echo -e "${RED}Error: File $input_dir/$cudart_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$cudart_file"

if [ ! -f "$input_dir/$cupti_file" ]; then
    echo -e "${RED}Error: File $input_dir/$cupti_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$cupti_file"

if [ ! -f "$input_dir/$json_file" ]; then
    echo -e "${RED}Error: File $input_dir/$json_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$json_file"

if [ ! -f "$input_dir/$googletest_file" ]; then
    echo -e "${RED}Error: File $input_dir/$googletest_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$googletest_file"

if [ ! -f "$input_dir/$triton_shared_file" ]; then
    echo -e "${YELLOW}Warning: File $input_dir/$triton_shared_file does not exist. This file is optional, please check if you need it.${NC}"
    triton_shared_file=""
else
    echo -e "Find $input_dir/$triton_shared_file"
fi

echo -e "cd ${input_dir}"
cd "$input_dir"

echo -e "Compressing..."
zip "$output_zip" "$nvcc_ptxas_file" "$nvcc_cudacrt_file" "$nvcc_ptxas_blackwell_file" "$cuobjdump_file" "$nvdisasm_file" "$cudart_file" \
    "$cupti_file" "$json_file" "$googletest_file" "$triton_shared_file"

echo -e "cd -"
cd -

echo -e ""
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Offline Build dependencies are successfully compressed into $output_zip${NC}"
    exit 0
else
    echo -e "${RED}Error: Failed to compress offline build dependencies${NC}"
    exit 1
fi
