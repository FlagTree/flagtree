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

nvcc_file="cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
cuobjdump_file="cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
nvdisam_file="cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
cudart_file="cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
cupti_file="cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
json_file="include.zip"
pybind11_file="pybind11-${pybind11_version}.tar.gz"
googletest_file="googletest-release-1.12.1.zip"
triton_shared_file="triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"

output_zip="offline-packed-nv${nv_toolchain_version}-pybind${pybind11_version}.zip"

if [ ! -d "$input_dir" ]; then
    echo "Error: offline build download directory $input_dir does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find ${nvcc_file}"

if [ ! -f "$input_dir/$cuobjdump_file" ]; then
    echo "Error: File $input_dir/$cuobjdump_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$cuobjdump_file"

if [ ! -f "$input_dir/$nvdisam_file" ]; then
    echo "Error: File $input_dir/$nvdisam_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$nvdisam_file"

if [ ! -f "$input_dir/$cudart_file" ]; then
    echo "Error: File $input_dir/$cudart_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$cudart_file"

if [ ! -f "$input_dir/$cupti_file" ]; then
    echo "Error: File $input_dir/$cupti_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$cupti_file"

if [ ! -f "$input_dir/$json_file" ]; then
    echo "Error: File $input_dir/$json_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$json_file"

if [ ! -f "$input_dir/$pybind11_file" ]; then
    echo "Error: File $input_dir/$pybind11_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$pybind11_file"

if [ ! -f "$input_dir/$googletest_file" ]; then
    echo "Error: File $input_dir/$googletest_file does not exist, run README_offline_build.sh for more information"
    exit 1
fi
echo "Find $input_dir/$googletest_file"

if [ ! -f "$input_dir/$triton_shared_file" ]; then
    echo "Warning: File $input_dir/$triton_shared_file does not exist. This file is optional, please check if you need it."
    triton_shared_file=""
else
    echo "Find $input_dir/$triton_shared_file"
fi

echo "cd ${input_dir}"
cd "$input_dir"

echo "Compressing..."
zip "$output_zip" "$nvcc_file" "$cuobjdump_file" "$nvdisam_file" "$cudart_file" "$cupti_file" \
    "$json_file" "$pybind11_file" "$googletest_file" "$triton_shared_file"

echo "cd -"
cd -

echo "mv $input_dir/$output_zip ."
mv $input_dir/$output_zip .

echo ""
if [ $? -eq 0 ]; then
    echo "Offline Build dependencies are successfully compressed into $output_zip"
    exit 0
else
    echo "Error: Failed to compress offline build dependencies"
    exit 1
fi
