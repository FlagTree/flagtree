#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

printfln() {
    printf "%b
" "$@"
}

printfln " =================== Offline Build README ==================="
SCRIPT_DIR=$(dirname $0)
# detect pybind11 version requirement
PYBIND11_VERSION_FILE="$SCRIPT_DIR/../../cmake/pybind11-version.txt"
if [ -f "$PYBIND11_VERSION_FILE" ]; then
    pybind11_version=$(tr -d '\n' < "$PYBIND11_VERSION_FILE")
    printfln "Pybind11 Version Required: $pybind11_version"
else
    printfln "${RED}Error: version file $PYBIND11_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="$SCRIPT_DIR/../../cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    printfln "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    printfln "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
fi

# handle system arch
if [ $# -eq 0 ]; then
    printfln "${RED}Error: No system architecture specified for offline build.${NC}"
    printfln "${GREEN}Usage: sh $0 arch=<system arch>${NC}"
    printfln "You need to specify the target system architecture to build the FlagTree"
    printfln "Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
    exit 1
fi

arch_param="$1"
case "$arch_param" in
    arch=*)
        arch="${arch_param#arch=}"
        ;;
    *)
        arch="$arch_param"
        ;;
esac

case "$arch" in
    x86_64)
        arch="64"
        ;;
    arm64|aarch64)
        arch="aarch64"
        ;;
    *)
        printfln "${RED}Error: Unsupported system architecture '$arch'.${NC}"
        printfln "${GREEN}Usage: sh $0 arch=<system arch>${NC}"
        printfln "Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
        exit 1
        ;;
esac
printfln "Target System Arch for offline building: $arch"

printfln ""
printfln "This is a guide for building FlagTree with default backend in an offline environment."
printfln ""
printfln "${YELLOW}>>>>> Step-1${NC} Download the dependencies according to the following methods:"
printfln "You can choose three download methods:"
printfln ""
printfln "  ${BLUE}1. Manually download:${NC}"
printfln "      NVCC should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-nvcc/${nv_toolchain_version}/download/linux-${arch}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
printfln "      CUOBJBDUMP should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-cuobjdump/${nv_toolchain_version}/download/linux-${arch}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
printfln "      NVDISAM should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-nvdisasm/${nv_toolchain_version}/download/linux-${arch}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
printfln "      CUDART should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-cudart-dev/${nv_toolchain_version}/download/linux-${arch}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
printfln "      CUPTI should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-cupti/${nv_toolchain_version}/download/linux-${arch}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
printfln "      JSON library should be downloaded from: ${BLUE}https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/include.zip"
printfln "      PYBIND11 should be downloaded from: ${BLUE}https://github.com/pybind/pybind11/archive/refs/tags/v${pybind11_version}.tar.gz${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/pybind11-${pybind11_version}.tar.gz"
printfln "      GOOGLETEST should be downloaded from ${BLUE}https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/googletest-release-1.12.1.zip"
printfln "      (TRITON_SHARED is optional):"
printfln "      TRITON_SHARED should be downloaded from: ${BLUE}https://github.com/microsoft/triton-shared/archive/380b87122c88af131530903a702d5318ec59bb33.zip${NC}"
printfln "          and stored as: <YOUR_DOWNLOAD_DIR>/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"
printfln ""
printfln "  ${BLUE}2. Use the script to download.${NC} You can specify the directory the store the downloaded files into:"
printfln "          ${GREEN}$ sh scripts/offline_build_download.sh arch=<system arch> <YOUR_DOWNLOAD_DIR>${NC}"
printfln ""
printfln "  ${BLUE}3. Directly download the packed file${NC}. Then you can jump to the ${GREEN}Step-3${NC}:"
printfln "          TODO: add the link to the .zip file"
printfln ""
printfln "${YELLOW}>>>>> Step-2${NC} Run the script to pack the dependencies into a .zip file. You can specify the source directory"
printfln "      providing the downloaded files and the output directory to store the packed .zip file"
printfln "       # Specify the input & output directory, the script will compress the files in YOU_DOWNLOAD_DIR"
printfln "       # into a .zip file in YOU_PACK_DIR"
printfln "          ${GREEN}$ sh scripts/offline_build_pack.sh <YOUR_DOWNLOAD_DIR> <YOUR_PACK_DIR>${NC}"
printfln ""
printfln "${YELLOW}>>>>> Step-3${NC} After uploading the packed .zip file to the offline environment, run the script scripts/offline_build_unpack.sh "
printfln "      to extract the dependencies to an appropriate location for FlagTree to copy. You can specify the directory to store the"
printfln "      packed .zip file and the directory to store the unpacked dependencies."
printfln "       # Specify the input & output directory, the script will extract the packed .zip file in YOUR_INPUT_DIR"
printfln "       # into the YOUR_UNPACK_DIR"
printfln "          ${GREEN}$ sh scripts/offline_build_unpack.sh <YOUR_INPUT_DIR> <YOUR_UNPACK_DIR>${NC}"
printfln ""
printfln "${YELLOW}>>>>> Step-4${NC} You can proceed with the installation normally according to the README.md."
printfln "      NOTE: Set the environment variables required for offline build before running 'pip install'"
printfln "            The FLAGTREE_OFFLINE_BUILD_DIR should be set to the ${BLUE}absolute path${NC} of the directory where the"
printfln "            unpacked dependencies are stored."
printfln "          ${GREEN}$ export TRITON_OFFLINE_BUILD=ON${NC}"
printfln "          ${GREEN}$ export FLAGTREE_OFFLINE_BUILD_DIR=<ABSOLUTE_PATH_OF_YOUR_UNPACK_DIR>${NC}"
printfln ""
printfln " =============================================="
