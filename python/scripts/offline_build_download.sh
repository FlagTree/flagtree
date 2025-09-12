#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Start Downloading Offline Build Files ==================="

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
    echo -e "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
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
        echo -e "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
        echo -e "   Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
        exit 1
        ;;
esac
echo -e "Target System Arch for offline building: $arch"

system="linux"
echo -e "Target System for offline building: $system"

check_download() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Download Success${NC}"
    else
        echo -e "${RED}Download Failed !!!${NC}"
        exit 1
    fi
    echo -e ""
}

if [ $# -ge 2 ]; then
    target_dir="$2"
    echo -e "${BLUE}Use $target_dir as download output directory${NC}"
else
    echo -e "${RED}Error: No output directory specified for downloading.${NC}"
    echo -e "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
    echo -e "   Support system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
    exit 1
fi

echo -e ""
if [ ! -d "$target_dir" ]; then
    echo -e "Creating download output directory $target_dir"
    mkdir -p "$target_dir"
else
    echo -e "Download output directory $target_dir already exists"
fi
echo -e ""

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

echo -e "Downloading PTXAS from: ${BLUE}$ptxas_url${NC}"
echo -e "wget $ptxas_url -O ${target_dir}/cuda-ptxas-${ptxas_version}-0.tar.bz2"
wget "$ptxas_url" -O ${target_dir}/cuda-ptxas-${ptxas_version}-0.tar.bz2
check_download

echo -e "Downloading CUDACRT from: ${BLUE}$cudacrt_url${NC}"
echo -e "wget $cudacrt_url -O ${target_dir}/cuda-crt-${cudacrt_version}-0.tar.bz2"
wget "$cudacrt_url" -O ${target_dir}/cuda-crt-${cudacrt_version}-0.tar.bz2
check_download

cuobjdump_url=https://anaconda.org/nvidia/cuda-cuobjdump/${cuobjdump_version}/download/linux-${arch}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2
echo -e "Downloading CUOBJBDUMP from: ${BLUE}$cuobjdump_url${NC}"
echo -e "wget $cuobjdump_url -O ${target_dir}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2"
wget "$cuobjdump_url" -O ${target_dir}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2
check_download

nvdisasm_url=https://anaconda.org/nvidia/cuda-nvdisasm/${nvdisasm_version}/download/linux-${arch}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2
echo -e "Downloading NVDISASM from: ${BLUE}$nvdisasm_url${NC}"
echo -e "wget $nvdisasm_url -O ${target_dir}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2"
wget "$nvdisasm_url" -O ${target_dir}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2
check_download

echo -e "Downloading CUDART from: ${BLUE}$cudart_url${NC}"
echo -e "wget $cudart_url -O ${target_dir}/cuda-cudart-dev-${cudart_version}-0.tar.bz2"
wget "$cudart_url" -O ${target_dir}/cuda-cudart-dev-${cudart_version}-0.tar.bz2
check_download

echo -e "Downloading CUPTI from: ${BLUE}$cupti_url${NC}"
echo -e "wget $cupti_url -O ${target_dir}/cuda-cupti-${cutpti_version}-0.tar.bz2"
wget "$cupti_url" -O ${target_dir}/cuda-cupti-${cupti_version}-0.tar.bz2
check_download

json_url=https://github.com/nlohmann/json/releases/download/${json_version}/include.zip
echo -e "Downloading JSON library from: ${BLUE}$json_url${NC}"
echo -e "wget $json_url -O ${target_dir}/include.zip"
wget "$json_url" -O ${target_dir}/include.zip
check_download

googletest_url=https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
echo -e "Downloading GoogleTest from: ${BLUE}$googletest_url${NC}"
echo -e "wget $googletest_url -O ${target_dir}/googletest-release-1.12.1.zip"
wget "$googletest_url" -O ${target_dir}/googletest-release-1.12.1.zip
check_download

triton_shared_url=https://github.com/microsoft/triton-shared/archive/380b87122c88af131530903a702d5318ec59bb33.zip
echo -e "Downloading Triton_Shared from: ${BLUE}$triton_shared_url${NC}"
echo -e "wget $triton_shared_url -O ${target_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"
wget "$triton_shared_url" -O ${target_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip
check_download

echo -e " =================== Done ==================="