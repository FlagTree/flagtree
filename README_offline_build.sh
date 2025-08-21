#!/bin/bash

echo " =================== Offline Build README ==================="
# detect pybind11 version requirement
PYBIND11_VERSION_FILE="cmake/pybind11-version.txt"
if [ -f "$PYBIND11_VERSION_FILE" ]; then
    pybind11_version=$(tr -d '\n' < "$PYBIND11_VERSION_FILE")
    # echo "Pybind11 Version Required: $pybind11_version"
else
    echo "Error: version file $PYBIND11_VERSION_FILE is not exist"
    exit 1
fi

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    # echo "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    echo "Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist"
    exit 1
fi

arch=$(uname -m)
# echo "Detected the system arch: $arch"

case "$arch" in
    x86_64)
        arch="64"
        ;;
    arm64|aarch64)
        arch="aarch64"
        ;;
    *)
        ;;
esac

echo ""
echo "This is a guide for building FlagTree in an offline environment."
echo ""
echo ">>>>> First, download the dependencies according to the following methods:"
echo "You can choose two download methods:"
echo ""
echo "  1. Manually download:"
echo "      NVCC should be downloaded from: https://anaconda.org/nvidia/cuda-nvcc/${nv_toolchain_version}/download/linux-${arch}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
echo "      CUOBJBDUMP should be downloaded from: https://anaconda.org/nvidia/cuda-cuobjdump/${nv_toolchain_version}/download/linux-${arch}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
echo "      NVDISAM should be downloaded from: https://anaconda.org/nvidia/cuda-nvdisasm/${nv_toolchain_version}/download/linux-${arch}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
echo "      CUDART should be downloaded from: https://anaconda.org/nvidia/cuda-cudart-dev/${nv_toolchain_version}/download/linux-${arch}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
echo "      CUPTI should be downloaded from: https://anaconda.org/nvidia/cuda-cupti/${nv_toolchain_version}/download/linux-${arch}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
echo "      JSON library should be downloaded from: https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/include.zip"
echo "      PYBIND11 should be downloaded from: https://github.com/pybind/pybind11/archive/refs/tags/v${pybind11_version}.tar.gz"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/pybind11-${pybind11_version}.tar.gz"
echo "      GOOGLETEST should be downloaded from https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/googletest-release-1.12.1.zip"
echo "      (TRITON_SHARED is optional):"
echo "      TRITON_SHARED should be downloaded from https://github.com/microsoft/triton-shared/archive/380b87122c88af131530903a702d5318ec59bb33.zip"
echo "          and stored as: <YOUR_DOWNLOAD_DIR>/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"
echo ""
echo "  2. Use the script to download. The default to store the downloaded files is $HOME/.flagtree-offline-download,"
echo "     you can specify the directory the store the downloaded files into: (default: $HOME/.flagtree-offline-download)"
echo "       # Default"
echo "          $ sh utils/offline_build_download.sh"
echo "       # Specify directory to store downloaded files"
echo "          $ sh utils/offline_build_download.sh <YOUR_DOWNLOAD_DIR>"
echo ""
echo ">>>>> Second, run the script to pack the dependencies into a .zip file. You can specify the source directory"
echo "      providing the downloaded files (default: $HOME/.flagtree-offline-download) and the output directory to"
echo "      store the packed .zip file (default: $PWD)"
echo "       # Default"
echo "          $ sh utils/offline_build_pack.sh"
echo "       # Specify the input & output directory, the script will compress the files in YOU_DOWNLOAD_DIR"
echo "       # into a .zip file in YOU_PACK_DIR"
echo "          $ sh utils/offline_build_pack.sh <YOUR_DOWNLOAD_DIR> <YOUR_PACK_DIR>"
echo ""
echo ">>>>> Third, after uploading the packed .zip file to the offline environment, run the script utils/offline_build_unpack.sh "
echo "      to extract the dependencies to an appropriate location for FlagTree to copy. You can specify the directory to store the"
echo "      packed .zip file (default: $PWD) and the directory to store the unpacked dependencies. (default: $HOME/.flagtree-offline-build)"
echo "       # Default"
echo "          $ sh utils/offline_build_unpack.sh"
echo "       # Specify the input & output directory, the script will extract the packed .zip file in YOUR_INPUT_DIR"
echo "       # into the YOUR_UNPACK_DIR"
echo "          $ sh utils/offline_build_unpack.sh <YOUR_INPUT_DIR> <YOUR_UNPACK_DIR>"
echo ""
echo ">>>>> Finally, you can proceed with the installation normally according to the README.md."
echo "      NOTE: Set the environment variables required for offline build before running 'pip install'"
echo "          $ export TRITON_OFFLLINE_BUILD=ON"
echo "          $ export FLAGTREE_OFFLINE_BUILD_DIR=<YOUR_UNPACK_DIR>"
echo ""
echo " =============================================="