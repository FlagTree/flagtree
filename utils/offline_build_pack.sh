#!/bin/bash

echo " =================== Start Packing Downloaded Offline Build Files ==================="
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

if [ $# -ge 1 ]; then
    input_dir="$1"
else
    input_dir="$HOME/.flagtree-offline-download"
fi

echo ""
if [ ! -d "$input_dir" ]; then
    echo "Creating default download directory $input_dir"
    mkdir -p "$input_dir"
else
    echo "Default download directory $input_dir already exists"
fi
echo ""

output_dir="$PWD"

nvcc_file="${input_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
cuobjdump_file="${input_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
nvdisam_file="${input_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
cudart_file="${input_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
cupti_file="${input_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
json_file="${input_dir}/include.zip"
pybind11_file="${input_dir}/pybind11-${pybind11_version}.tar.gz"
googletest_file="${input_dir}/googletest-release-1.12.1.zip"
triton_shared_file="${input_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"

output_zip="./offline-packed-nv${nv_toolchain_version}-pybind${pybind11_version}.zip"

if [ ! -d "$input_dir" ]; then
    echo "Error: offline build download directory $input_dir does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${nvcc_file}"

if [ ! -f "$cuobjdump_file" ]; then
    echo "Error: File $cuobjdump_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${cuobjdump_file}"

if [ ! -f "$nvdisam_file" ]; then
    echo "Error: File $nvdisam_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${nvdisam_file}"

if [ ! -f "$cudart_file" ]; then
    echo "Error: File $cudart_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${cudart_file}"

if [ ! -f "$cupti_file" ]; then
    echo "Error: File $cupti_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${cupti_file}"

if [ ! -f "$json_file" ]; then
    echo "Error: File $json_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${json_file}"

if [ ! -f "$pybind11_file" ]; then
    echo "Error: File $pybind11_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${pybind11_file}"

if [ ! -f "$googletest_file" ]; then
    echo "Error: File $googletest_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${googletest_file}"

if [ ! -f "$triton_shared_file" ]; then
    echo "Warning: File $triton_shared_file does not exist. This file is optional, please check if you need it."
    triton_shared_file=""
else
    echo "Find ${triton_shared_file}"
fi

echo "Compressing..."
zip "$output_zip" "$nvcc_file" "$cuobjdump_file" "$nvdisam_file" "$cudart_file" "$cupti_file" \
    "$json_file" "$pybind11_file" "$googletest_file" "$triton_shared_file"

echo ""
if [ $? -eq 0 ]; then
    echo "Offline Build dependencies are successfully compressed into $output_zip"
    exit 0
else
    echo "Error: Failed to compress offline build dependencies"
    exit 1
fi
