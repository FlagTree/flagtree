#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

printfln() {
    printf "%b
" "$@"
}

printfln " =================== Start Downloading Offline Build Files ==================="

# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="../cmake/nvidia-toolchain-version.json"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    ptxas_version=$(grep '"ptxas"' "$NV_TOOLCHAIN_VERSION_FILE" | grep -v "ptxas-blackwell" | sed -E 's/.*"ptxas": "([^"]+)".*/\1/')
    cuobjdump_version=$(grep '"cuobjdump"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cuobjdump": "([^"]+)".*/\1/')
    nvdisasm_version=$(grep '"nvdisasm"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"nvdisasm": "([^"]+)".*/\1/')
    cudacrt_version=$(grep '"cudacrt"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudacrt": "([^"]+)".*/\1/')
    cudart_version=$(grep '"cudart"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudart": "([^"]+)".*/\1/')
    cupti_version=$(grep '"cupti"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cupti": "([^"]+)".*/\1/')
    printfln "Nvidia Toolchain Version Requirement:"
    printfln "   ptxas: $ptxas_version"
    printfln "   cuobjdump: $cuobjdump_version"
    printfln "   nvdisasm: $nvdisasm_version"
    printfln "   cudacrt: $cudacrt_version"
    printfln "   cudart: $cudart_version"
    printfln "   cupti: $cupti_version"
else
    printfln "${RED}Error: version file $NV_TOOLCHAIN_VERSION_FILE is not exist${NC}"
    exit 1
fi

# detect json version requirement
JSON_VERSION_FILE="../cmake/json-version.txt"
if [ -f "$JSON_VERSION_FILE" ]; then
    json_version=$(tr -d '\n' < "$JSON_VERSION_FILE")
    printfln "JSON Version Required: $json_version"
else
    printfln "${RED}Error: version file $JSON_VERSION_FILE is not exist${NC}"
fi

# handle system arch
if [ $# -eq 0 ]; then
    printfln "${RED}Error: No system architecture specified for offline build.${NC}"
    printfln "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
    printfln "You need to specify the target system architecture to build the FlagTree"
    printfln "Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
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
        printfln "${RED}Error: Unsupported system architecture '$arch'.${NC}"
        printfln "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
        printfln "   Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
        exit 1
        ;;
esac
printfln "Target System Arch for offline building: $arch"

system="linux"
printfln "Target System for offline building: $system"

check_download() {
    if [ $? -eq 0 ]; then
        printfln "${GREEN}Download Success${NC}"
    else
        printfln "${RED}Download Failed !!!${NC}"
        exit 1
    fi
    printfln ""
}

if [ $# -ge 2 ]; then
    target_dir="$2"
    printfln "${BLUE}Use $target_dir as download output directory${NC}"
else
    printfln "${RED}Error: No output directory specified for downloading.${NC}"
    printfln "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
    printfln "   Support system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
    exit 1
fi

printfln ""
if [ ! -d "$target_dir" ]; then
    printfln "Creating download output directory $target_dir"
    mkdir -p "$target_dir"
else
    printfln "Download output directory $target_dir already exists"
fi
printfln ""

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

printfln "Downloading PTXAS from: ${BLUE}$ptxas_url${NC}"
printfln "wget $ptxas_url -O ${target_dir}/cuda-ptxas-${ptxas_version}-0.tar.bz2"
wget "$ptxas_url" -O ${target_dir}/cuda-ptxas-${ptxas_version}-0.tar.bz2
check_download

printfln "Downloading CUDACRT from: ${BLUE}$cudacrt_url${NC}"
printfln "wget $cudacrt_url -O ${target_dir}/cuda-crt-${cudacrt_version}-0.tar.bz2"
wget "$cudacrt_url" -O ${target_dir}/cuda-crt-${cudacrt_version}-0.tar.bz2
check_download

cuobjdump_url=https://anaconda.org/nvidia/cuda-cuobjdump/${cuobjdump_version}/download/linux-${arch}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2
printfln "Downloading CUOBJBDUMP from: ${BLUE}$cuobjdump_url${NC}"
printfln "wget $cuobjdump_url -O ${target_dir}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2"
wget "$cuobjdump_url" -O ${target_dir}/cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2
check_download

nvdisasm_url=https://anaconda.org/nvidia/cuda-nvdisasm/${nvdisasm_version}/download/linux-${arch}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2
printfln "Downloading NVDISASM from: ${BLUE}$nvdisasm_url${NC}"
printfln "wget $nvdisasm_url -O ${target_dir}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2"
wget "$nvdisasm_url" -O ${target_dir}/cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2
check_download

printfln "Downloading CUDART from: ${BLUE}$cudart_url${NC}"
printfln "wget $cudart_url -O ${target_dir}/cuda-cudart-dev-${cudart_version}-0.tar.bz2"
wget "$cudart_url" -O ${target_dir}/cuda-cudart-dev-${cudart_version}-0.tar.bz2
check_download

printfln "Downloading CUPTI from: ${BLUE}$cupti_url${NC}"
printfln "wget $cupti_url -O ${target_dir}/cuda-cupti-${cutpti_version}-0.tar.bz2"
wget "$cupti_url" -O ${target_dir}/cuda-cupti-${cupti_version}-0.tar.bz2
check_download

json_url=https://github.com/nlohmann/json/releases/download/${json_version}/include.zip
printfln "Downloading JSON library from: ${BLUE}$json_url${NC}"
printfln "wget $json_url -O ${target_dir}/include.zip"
wget "$json_url" -O ${target_dir}/include.zip
check_download

googletest_url=https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
printfln "Downloading GoogleTest from: ${BLUE}$googletest_url${NC}"
printfln "wget $googletest_url -O ${target_dir}/googletest-release-1.12.1.zip"
wget "$googletest_url" -O ${target_dir}/googletest-release-1.12.1.zip
check_download

triton_ascend_url=https://gitee.com/ascend/triton-ascend/repository/archive/master.zip
ascendnpu_ir_url=https://gitee.com/ascend/ascendnpu-ir/repository/archive/1922371c42749fda534d6395b7ed828b5c9f36d4.zip
triton_url=https://github.com/triton-lang/triton/archive/9641643da6c52000c807b5eeed05edaec4402a67.zip
printfln "Downloading Triton_Ascend from: ${BLUE}$triton_ascend_url${NC}"
printfln "wget $triton_ascend_url -O ${target_dir}/triton-ascend-master.zip"
wget "$triton_ascend_url" -O ${target_dir}/triton-ascend-master.zip
check_download
printfln "Downloading AscendNPU IR for Triton_Ascend from: ${BLUE}$ascendnpu_ir_url${NC}"
printfln "wget $ascendnpu_ir_url -O ${target_dir}/ascendnpu-ir-1922371c42749fda534d6395b7ed828b5c9f36d4.zip"
wget "$ascendnpu_ir_url" -O ${target_dir}/ascendnpu-ir-1922371c42749fda534d6395b7ed828b5c9f36d4.zip
check_download
printfln "Downloading Triton for Triton_Ascend from: ${BLUE}$triton_url${NC}"
printfln "wget $triton_url -O ${target_dir}/triton-9641643da6c52000c807b5eeed05edaec4402a67.zip"
wget "$triton_url" -O ${target_dir}/triton-9641643da6c52000c807b5eeed05edaec4402a67.zip
check_download

triton_shared_url=https://github.com/microsoft/triton-shared/archive/380b87122c88af131530903a702d5318ec59bb33.zip
printfln "Downloading Triton_Shared from: ${BLUE}$triton_shared_url${NC}"
printfln "wget $triton_shared_url -O ${target_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"
wget "$triton_shared_url" -O ${target_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip
check_download

printfln " =================== Done ==================="
