#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Start Unpacking Offline Build Dependencies ==================="
echo -e ""
# detect pybind11 version requirement
PYBIND11_VERSION_FILE="cmake/pybind11-version.txt"
if [ -f "$PYBIND11_VERSION_FILE" ]; then
    pybind11_version=$(tr -d '\n' < "$PYBIND11_VERSION_FILE")
    echo -e "Pybind11 Version Required: $pybind11_version"
else
    echo -e "${RED}Error: version file $PYBIND11_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    echo -e "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    echo -e "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
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



echo -e "Unpacking ${input_zip} into ${output_dir}..."
unzip "${input_zip}" -d ${output_dir}

echo -e "Creating directory ${output_dir}/nvidia ..."
mkdir "${output_dir}/nvidia"

echo -e "Creating directory ${output_dir}/nvidia/ptxas ..."
mkdir "${output_dir}/nvidia/ptxas"
echo -e "Extracting $nvcc_file into ${output_dir}/nvidia/ptxas ..."
tar -jxf $nvcc_file -C "${output_dir}/nvidia/ptxas"

echo -e "Creating directory ${output_dir}/nvidia/cuobjdump ..."
mkdir "${output_dir}/nvidia/cuobjdump"
echo -e "Extracting $cuobjdump_file into ${output_dir}/nvidia/cuobjdump ..."
tar -jxf $cuobjdump_file -C "${output_dir}/nvidia/cuobjdump"

echo -e "Creating directory ${output_dir}/nvidia/nvdisasm ..."
mkdir "${output_dir}/nvidia/nvdisasm"
echo -e "Extracting $nvdisam_file into ${output_dir}/nvidia/nvdisasm ..."
tar -jxf $nvdisam_file -C "${output_dir}/nvidia/nvdisasm"

echo -e "Creating directory ${output_dir}/nvidia/cudacrt ..."
mkdir "${output_dir}/nvidia/cudacrt"
echo -e "Extracting $nvcc_file into ${output_dir}/nvidia/cudacrt ..."
tar -jxf $nvcc_file -C "${output_dir}/nvidia/cudacrt"

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

echo -e "Creating directory ${output_dir}/pybind11 ..."
mkdir "${output_dir}/pybind11"
echo -e "Extracting $pybind11_file into ${output_dir}/pybind11 ..."
tar -zxf $pybind11_file -C "${output_dir}/pybind11"

echo -e "Extracting $googletest_file into ${output_dir}/googletest-release-1.12.1 ..."
unzip $googletest_file -d "${output_dir}" > /dev/null

if [ -f "${triton_shared_file}" ]; then
    echo -e "Extracting $triton_shared_file into ${output_dir}/triton_shared ..."
    unzip $triton_shared_file -d "${output_dir}" > /dev/null
    mv ${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33 ${output_dir}/triton_shared
else
    echo -e "Warning: File $triton_shared_file does not exist. This file is optional, please check if you need it."
fi

echo -e ""
echo -e "Delete $nvcc_file"
rm $nvcc_file
echo -e "Delete $cuobjdump_file"
rm $cuobjdump_file
echo -e "Delete $nvdisam_file"
rm $nvdisam_file
echo -e "Delete $cudart_file"
rm $cudart_file
echo -e "Delete $cupti_file"
rm $cupti_file
echo -e "Delete $json_file"
rm $json_file
echo -e "Delete $pybind11_file"
rm $pybind11_file
echo -e "Delete $googletest_file"
rm $googletest_file
if [ -f "${triton_shared_file}" ]; then
    echo -e "Delete $triton_shared_file"
    rm $triton_shared_file
fi
echo -e "Delete useless file: ${output_dir}/nvidia/cudart/lib/libcudart.so"
rm ${output_dir}/nvidia/cudart/lib/libcudart.so