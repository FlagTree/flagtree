#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Start Packing Downloaded Offline Build Files ==================="
echo -e ""

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="../cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    echo -e "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    echo -e "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
fi

output_zip="offline-packed-nv${nv_toolchain_version}.zip"

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

nvcc_file="cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
cuobjdump_file="cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
nvdisam_file="cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
cudart_file="cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
cupti_file="cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
json_file="include.zip"
googletest_file="googletest-release-1.12.1.zip"
triton_shared_file="triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"

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

if [ ! -f "$input_dir/$nvdisam_file" ]; then
    echo -e "${RED}Error: File $input_dir/$nvdisam_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
echo -e "Find $input_dir/$nvdisam_file"

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
    echo -e "Warning: File $input_dir/$triton_shared_file does not exist. This file is optional, please check if you need it."
    triton_shared_file=""
else
    echo -e "Find $input_dir/$triton_shared_file"
fi

echo -e "cd ${input_dir}"
cd "$input_dir"

echo -e "Compressing..."
zip "$output_zip" "$nvcc_file" "$cuobjdump_file" "$nvdisam_file" "$cudart_file" "$cupti_file" \
    "$json_file" "$googletest_file" "$triton_shared_file"

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
