#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Offline Build README ==================="

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

# handle system arch
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No system architecture specified for offline build.${NC}"
    echo -e "${GREEN}Usage: sh $0 arch=<system arch>${NC}"
    echo -e "You need to specify the target system architecture to build the FlagTree"
    echo -e "Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
    exit 1
fi

arch_param="$1"
if [[ "$arch_param" == arch=* ]]; then
    arch="${arch_param#arch=}"
else
    arch="$arch_param"
fi

case "$arch" in
    x86_64)
        arch="64"
        ;;
    arm64|aarch64)
        arch="aarch64"
        ;;
    *)
        echo -e "${RED}Error: Unsupported system architecture '$arch'.${NC}"
        echo -e "${GREEN}Usage: sh $0 arch=<system arch>${NC}"
        echo -e "Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
        exit 1
        ;;
esac
echo -e "Target System Arch for offline building: $arch"

system="linux"
echo -e "Target System for offline building: $system"

# generate download URLs
version_major=$(echo $ptxas_version | cut -d. -f1)
version_minor1=$(echo $ptxas_version | cut -d. -f2)
if [ "$version_major" -ge 12 ] && [ "$version_minor1" -ge 5 ]; then
    ptxas_url="https://anaconda.org/nvidia/cuda-nvcc-tools/${ptxas_version}/download/${system}-${arch}/cuda-nvcc-tools-${ptxas_version}-0.tar.bz2"
else
    ptxas_url="https://anaconda.org/nvidia/cuda-nvcc/${ptxas_version}/download/${system}-${arch}/cuda-nvcc-${ptxas_version}-0.tar.bz2"
fi

version_major=$(echo $cudacrt_version | cut -d. -f1)
version_minor1=$(echo $cudacrt_version | cut -d. -f2)
if [ "$version_major" -ge 12 ] && [ "$version_minor1" -ge 5 ]; then
    cudacrt_url="https://anaconda.org/nvidia/cuda-crt-dev_${system}-${arch}/${cudacrt_version}/download/noarch/cuda-crt-dev_${system}-${arch}-${cudacrt_version}-0.tar.bz2"
else
    cudacrt_url="https://anaconda.org/nvidia/cuda-nvcc/${cudacrt_version}/download/${system}-${arch}/cuda-nvcc-${cudacrt_version}-0.tar.bz2"
fi

version_major=$(echo $cudart_version | cut -d. -f1)
version_minor1=$(echo $cudart_version | cut -d. -f2)
if [ "$version_major" -ge 12 ] && [ "$version_minor1" -ge 5 ]; then
    cudart_url="https://anaconda.org/nvidia/cuda-cudart-dev_${system}-${arch}/${cudart_version}/download/noarch/cuda-cudart-dev_${system}-${arch}-${cudart_version}-0.tar.bz2"
else
    cudart_url="https://anaconda.org/nvidia/cuda-cudart-dev/${cudart_version}/download/${system}-${arch}/cuda-cudart-dev-${cudart_version}-0.tar.bz2"
fi

version_major=$(echo $cupti_version | cut -d. -f1)
version_minor1=$(echo $cupti_version | cut -d. -f2)
if [ "$version_major" -ge 12 ] && [ "$version_minor1" -ge 5 ]; then
    cupti_url="https://anaconda.org/nvidia/cuda-cupti-dev/${cupti_version}/download/${system}-${arch}/cuda-cupti-dev-${cupti_version}-0.tar.bz2"
else
    cupti_url="https://anaconda.org/nvidia/cuda-cupti/${cupti_version}/download/${system}-${arch}/cuda-cupti-${cupti_version}-0.tar.bz2"
fi

echo -e ""
echo -e "This is a guide for building FlagTree with default backend in an offline environment."
echo -e ""
echo -e "${YELLOW}>>>>> Step-1${NC} Download the dependencies according to the following methods:"
echo -e "You can choose three download methods:"
echo -e ""
echo -e "  ${BLUE}1. Manually download:${NC}"
echo -e "      PTXAS should be downloaded from: ${BLUE}${ptxas_url}${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-ptxas-${ptxas_version}-0.tar.bz2"
echo -e "      CUDACRT should be downloaded from: ${BLUE}${cudacrt_url}${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-crt-${cudacrt_version}-0.tar.bz2"
echo -e "      CUOBJBDUMP should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-cuobjdump/${cuobjdump_version}/download/linux-${arch}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2"
echo -e "      NVDISASM should be downloaded from: ${BLUE}https://anaconda.org/nvidia/cuda-nvdisasm/${nvdisasm_version}/download/linux-${arch}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2"
echo -e "      CUDART should be downloaded from: ${BLUE}${cudart_url}${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cudart-dev-${cudart_version}-0.tar.bz2"
echo -e "      CUPTI should be downloaded from: ${BLUE}${cupti_url}${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/cuda-cupti-${cupti_version}-0.tar.bz2"
echo -e "      JSON library should be downloaded from: ${BLUE}https://github.com/nlohmann/json/releases/download/${json_version}/include.zip${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/include.zip"
echo -e "      GOOGLETEST should be downloaded from ${BLUE}https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/googletest-release-1.12.1.zip"
echo -e "      (TRITON_SHARED is optional):"
echo -e "      TRITON_SHARED should be downloaded from: ${BLUE}https://github.com/microsoft/triton-shared/archive/380b87122c88af131530903a702d5318ec59bb33.zip${NC}"
echo -e "          and stored as: <YOUR_DOWNLOAD_DIR>/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"
echo -e ""
echo -e "  ${BLUE}2. Use the script to download.${NC} You can specify the directory the store the downloaded files into:"
echo -e "          ${GREEN}$ sh scripts/offline_build_download.sh arch=<system arch> <YOUR_DOWNLOAD_DIR>${NC}"
echo -e ""
echo -e "  ${BLUE}3. Directly download the packed file${NC}. Then you can jump to the ${GREEN}Step-3${NC}:"
echo -e "          TODO: add the link to the .zip file"
echo -e ""
echo -e "${YELLOW}>>>>> Step-2${NC} Run the script to pack the dependencies into a .zip file. You can specify the source directory"
echo -e "      providing the downloaded files and the output directory to store the packed .zip file"
echo -e "       # Specify the input & output directory, the script will compress the files in YOU_DOWNLOAD_DIR"
echo -e "       # into a .zip file in YOU_PACK_DIR"
echo -e "          ${GREEN}$ sh scripts/offline_build_pack.sh <YOUR_DOWNLOAD_DIR> <YOUR_PACK_DIR>${NC}"
echo -e ""
echo -e "${YELLOW}>>>>> Step-3${NC} After uploading the packed .zip file to the offline environment, run the script scripts/offline_build_unpack.sh "
echo -e "      to extract the dependencies to an appropriate location for FlagTree to copy. You can specify the directory to store the"
echo -e "      packed .zip file and the directory to store the unpacked dependencies."
echo -e "       # Specify the input & output directory, the script will extract the packed .zip file in YOUR_INPUT_DIR"
echo -e "       # into the YOUR_UNPACK_DIR"
echo -e "          ${GREEN}$ sh scripts/offline_build_unpack.sh <YOUR_INPUT_DIR> <YOUR_UNPACK_DIR>${NC}"
echo -e ""
echo -e "${YELLOW}>>>>> Step-4${NC} You can proceed with the installation normally according to the README.md."
echo -e "      NOTE: Set the environment variables required for offline build before running 'pip install'"
echo -e "            The FLAGTREE_OFFLINE_BUILD_DIR should be set to the ${BLUE}absolute path${NC} of the directory where the"
echo -e "            unpacked dependencies are stored."
echo -e "          ${GREEN}$ export TRITON_OFFLLINE_BUILD=ON${NC}"
echo -e "          ${GREEN}$ export FLAGTREE_OFFLINE_BUILD_DIR=<ABSOLUTE_PATH_OF_YOUR_UNPACK_DIR>${NC}"
echo -e ""
echo -e " =============================================="