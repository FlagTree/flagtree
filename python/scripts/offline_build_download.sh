#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e " =================== Start Downloading Offline Build Files ==================="
# detect pybind11 version requirement
PYBIND11_VERSION_FILE="../cmake/pybind11-version.txt"
if [ -f "$PYBIND11_VERSION_FILE" ]; then
    pybind11_version=$(tr -d '\n' < "$PYBIND11_VERSION_FILE")
    echo -e "Pybind11 Version Required: $pybind11_version"
else
    echo -e "${RED}Error: version file $PYBIND11_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="../cmake/nvidia-toolchain-version.txt"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    nv_toolchain_version=$(tr -d '\n' < "$NV_TOOLCHAIN_VERSION_FILE")
    echo -e "Nvidia Toolchain Version Required: $nv_toolchain_version"
else
    echo -e "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
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

nvcc_url=https://anaconda.org/nvidia/cuda-nvcc/${nv_toolchain_version}/download/linux-${arch}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2
echo -e "Downloading NVCC from: ${BLUE}$nvcc_url${NC}"
echo -e "wget $nvcc_url -O ${target_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
wget "$nvcc_url" -O ${target_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2
check_download

cuobjdump_url=https://anaconda.org/nvidia/cuda-cuobjdump/${nv_toolchain_version}/download/linux-${arch}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2
echo -e "Downloading CUOBJBDUMP from: ${BLUE}$cuobjdump_url${NC}"
echo -e "wget $cuobjdump_url -O ${target_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
wget "$cuobjdump_url" -O ${target_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2
check_download

nvdisam_url=https://anaconda.org/nvidia/cuda-nvdisasm/${nv_toolchain_version}/download/linux-${arch}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2
echo -e "Downloading NVDISAM from: ${BLUE}$nvdisam_url${NC}"
echo -e "wget $nvdisam_url -O ${target_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
wget "$nvdisam_url" -O ${target_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2
check_download

cudart_url=https://anaconda.org/nvidia/cuda-cudart-dev/${nv_toolchain_version}/download/linux-${arch}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2
echo -e "Downloading CUDART from: ${BLUE}$cudart_url${NC}"
echo -e "wget $cudart_url -O ${target_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
wget "$cudart_url" -O ${target_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2
check_download

cupti_url=https://anaconda.org/nvidia/cuda-cupti/${nv_toolchain_version}/download/linux-${arch}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2
echo -e "Downloading CUPTI from: ${BLUE}$cupti_url${NC}"
echo -e "wget $cupti_url -O ${target_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
wget "$cupti_url" -O ${target_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2
check_download

pybind11_url=https://github.com/pybind/pybind11/archive/refs/tags/v${pybind11_version}.tar.gz
echo -e "Downloading Pybind11 from: ${BLUE}$pybind11_url${NC}"
echo -e "wget $pybind11_url -O ${target_dir}/pybind11-${pybind11_version}.tar.gz"
wget "$pybind11_url" -O ${target_dir}/pybind11-${pybind11_version}.tar.gz
check_download

json_url=https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
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
