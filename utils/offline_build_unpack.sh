#!/bin/bash

echo " =================== Start Preparing Offline Build Dependencies ==================="
echo ""
# detect pybind11 version requirement
PYBIND11_VERSION_FILE="cmake/pybind11-version.txt"
if [ -f "$PYBIND11_VERSION_FILE" ]; then
    pybind11_version=$(tr -d '\n' < "$PYBIND11_VERSION_FILE")
    echo "Pybind11 Version Required: $pybind11_version"
else
    echo "Error: version file $PYBIND11_VERSION_FILE is not exist"
    exit 1
fi

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    echo "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    echo "Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist"
    exit 1
fi

input_dir="$PWD"

if [ $# -ge 1 ]; then
    output_dir="$1"
else
    output_dir="$HOME/.flagtree-offline-build"
fi

echo ""
if [ ! -d "$output_dir" ]; then
    echo "Creating default download directory $output_dir"
    mkdir -p "$output_dir"
else
    echo "Default download directory $output_dir already exists"
fi
echo ""

nvcc_file="${output_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
cuobjdump_file="${output_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
nvdisam_file="${output_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
cudart_file="${output_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
cupti_file="${output_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
json_file="${output_dir}/include.zip"
pybind11_file="${output_dir}/pybind11-${pybind11_version}.tar.gz"
googletest_file="${output_dir}/googletest-release-1.12.1.zip"
triton_shared_file="${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"

zip_file="offline-packed-nv${nv_toolchain_version}-pybind${pybind11_version}.zip"

if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
fi

if [ ! -f "${input_dir}/${zip_file}" ]; then
    echo "Cannot find $zip_file in directory: $input_dir"
    exit 1
fi

echo "Unpacking $zip_file into ${output_dir}..."
unzip "${input_dir}/${zip_file}" -d ${output_dir}

echo "Creating directory ${output_dir}/nvidia..."
mkdir "${output_dir}/nvidia"

echo "Creating directory ${output_dir}/nvidia/ptxas ..."
mkdir "${output_dir}/nvidia/ptxas"
echo "Extracting $nvcc_file into ${output_dir}/nvidia/ptxas ..."
tar -jxf $nvcc_file -C "${output_dir}/nvidia/ptxas"

echo "Creating directory ${output_dir}/nvidia/cuobjdump ..."
mkdir "${output_dir}/nvidia/cuobjdump"
echo "Extracting $cuobjdump_file into ${output_dir}/nvidia/cuobjdump ..."
tar -jxf $cuobjdump_file -C "${output_dir}/nvidia/cuobjdump"

echo "Creating directory ${output_dir}/nvidia/nvdisasm ..."
mkdir "${output_dir}/nvidia/nvdisasm"
echo "Extracting $nvdisam_file into ${output_dir}/nvidia/nvdisasm ..."
tar -jxf $nvdisam_file -C "${output_dir}/nvidia/nvdisasm"

echo "Creating directory ${output_dir}/nvidia/cudacrt ..."
mkdir "${output_dir}/nvidia/cudacrt"
echo "Extracting $nvcc_file into ${output_dir}/nvidia/cudacrt ..."
tar -jxf $nvcc_file -C "${output_dir}/nvidia/cudacrt"

echo "Creating directory ${output_dir}/nvidia/cudart ..."
mkdir "${output_dir}/nvidia/cudart"
echo "Extracting $cudart_file into ${output_dir}/nvidia/cudart ..."
tar -jxf $cudart_file -C "${output_dir}/nvidia/cudart"

echo "Creating directory ${output_dir}/nvidia/cupti ..."
mkdir "${output_dir}/nvidia/cupti"
echo "Extracting $cupti_file into ${output_dir}/nvidia/cupti ..."
tar -jxf $cupti_file -C "${output_dir}/nvidia/cupti"

echo "Creating directory ${output_dir}/json ..."
mkdir "${output_dir}/json"
echo "Extracting $json_file into ${output_dir}/json ..."
unzip $json_file -d "${output_dir}/json" > /dev/null

echo "Creating directory ${output_dir}/pybind11 ..."
mkdir "${output_dir}/pybind11"
echo "Extracting $pybind11_file into ${output_dir}/pybind11 ..."
tar -zxf $pybind11_file -C "${output_dir}/pybind11"

echo "Extracting $googletest_file into ${output_dir}/googletest-release-1.12.1 ..."
unzip $googletest_file -d "${output_dir}" > /dev/null

if [ -f "${triton_shared_file}" ]; then
    echo "Extracting $triton_shared_file into ${output_dir}/triton-shared ..."
    unzip $triton_shared_file -d "${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33" > /dev/null
    mv ${output_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33 ${output_dir}/triton-shared
else
    echo "Warning: File $triton_shared_file does not exist. This file is optional, please check if you need it."
fi

echo ""
echo "Delete $nvcc_file"
rm $nvcc_file
echo "Delete $cuobjdump_file"
rm $cuobjdump_file
echo "Delete $nvdisam_file"
rm $nvdisam_file
echo "Delete $cudart_file"
rm $cudart_file
echo "Delete $cupti_file"
rm $cupti_file
echo "Delete $json_file"
rm $json_file
echo "Delete $pybind11_file"
rm $pybind11_file
echo "Delete $googletest_file"
rm $googletest_file
if [ ! -f "${triton_shared_file}" ]; then
    echo "Delete $triton_shared_file"
    rm $$triton_shared_file
fi