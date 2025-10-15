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
SCRIPT_DIR=$(dirname $0)
# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="$SCRIPT_DIR/../../cmake/nvidia-toolchain-version.json"
if [ -f "$NV_TOOLCHAIN_VERSION_FILE" ]; then
    ptxas_blackwell_version=$(grep '"ptxas-blackwell"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"ptxas-blackwell": "([^"]+)".*/\1/')
    ptxas_version=$(grep '"ptxas"' "$NV_TOOLCHAIN_VERSION_FILE" | grep -v "ptxas-blackwell" | sed -E 's/.*"ptxas": "([^"]+)".*/\1/')
    cuobjdump_version=$(grep '"cuobjdump"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cuobjdump": "([^"]+)".*/\1/')
    nvdisasm_version=$(grep '"nvdisasm"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"nvdisasm": "([^"]+)".*/\1/')
    cudacrt_version=$(grep '"cudacrt"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudacrt": "([^"]+)".*/\1/')
    cudart_version=$(grep '"cudart"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cudart": "([^"]+)".*/\1/')
    cupti_version=$(grep '"cupti"' "$NV_TOOLCHAIN_VERSION_FILE" | sed -E 's/.*"cupti": "([^"]+)".*/\1/')
    printfln "Nvidia Toolchain Version Requirement:"
    printfln "   ptxas: $ptxas_version"
    printfln "   ptxas-blackwell: $ptxas_blackwell_version"
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
JSON_VERSION_FILE="$SCRIPT_DIR/../../cmake/json-version.txt"
if [ -f "$JSON_VERSION_FILE" ]; then
    json_version=$(tr -d '\n' < "$JSON_VERSION_FILE")
    printfln "JSON Version Required: $json_version"
else
    printfln "${RED}Error: version file $JSON_VERSION_FILE is not exist${NC}"
    exit 1
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
        arch="x86_64"
        ;;
    arm64|aarch64)
        arch="sbsa"
        ;;
    *)
        printfln "${RED}Error: Unsupported system architecture '$arch'.${NC}"
        printfln "${GREEN}Usage: sh $0 arch=<system arch> <output_dir>${NC}"
        printfln "   Supported system arch values: ${GREEN}x86_64, arm64, aarch64${NC}"
        exit 1
        ;;
esac
printfln "Target System Arch for offline building: $arch"

# Only support linux currently
system="linux"

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

nvcc_ptxas_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/${system}-${arch}/cuda_nvcc-${system}-${arch}-${ptxas_version}-archive.tar.xz
printfln "Downloading NVCC ptxas from: ${BLUE}$nvcc_ptxas_url${NC}"
printfln "wget $nvcc_ptxas_url -O ${target_dir}/cuda-nvcc-${ptxas_version}.tar.xz"
wget "$nvcc_ptxas_url" -O ${target_dir}/cuda-nvcc-${ptxas_version}.tar.xz
check_download

nvcc_ptxas_blackwell_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/${system}-${arch}/cuda_nvcc-${system}-${arch}-${ptxas_blackwell_version}-archive.tar.xz
printfln "Downloading NVCC ptxas-blackwell from: ${BLUE}$nvcc_ptxas_blackwell_url${NC}"
printfln "wget $nvcc_ptxas_blackwell_url -O ${target_dir}/cuda-nvcc-${ptxas_blackwell_version}.tar.xz"
wget "$nvcc_ptxas_blackwell_url" -O ${target_dir}/cuda-nvcc-${ptxas_blackwell_version}.tar.xz
check_download

nvcc_cudacrt_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/${system}-${arch}/cuda_nvcc-${system}-${arch}-${cudacrt_version}-archive.tar.xz
printfln "Downloading NVCC cudacrt from: ${BLUE}$nvcc_cudacrt_url${NC}"
printfln "wget $nvcc_cudacrt_url -O ${target_dir}/cuda-nvcc-${cudacrt_version}.tar.xz"
wget "$nvcc_cudacrt_url" -O ${target_dir}/cuda-nvcc-${cudacrt_version}.tar.xz
check_download

cuobjdump_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/${system}-${arch}/cuda_cuobjdump-${system}-${arch}-${cuobjdump_version}-archive.tar.xz
printfln "Downloading CUOBJBDUMP from: ${BLUE}$cuobjdump_url${NC}"
printfln "wget $cuobjdump_url -O ${target_dir}/cuda-cuobjdump-${cuobjdump_version}.tar.xz"
wget "$cuobjdump_url" -O ${target_dir}/cuda-cuobjdump-${cuobjdump_version}.tar.xz
check_download

nvdisasm_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvdisasm/${system}-${arch}/cuda_nvdisasm-${system}-${arch}-${nvdisasm_version}-archive.tar.xz
printfln "Downloading NVDISASM from: ${BLUE}$nvdisasm_url${NC}"
printfln "wget $nvdisasm_url -O ${target_dir}/cuda-nvdisasm-${nvdisasm_version}.tar.xz"
wget "$nvdisasm_url" -O ${target_dir}/cuda-nvdisasm-${nvdisasm_version}.tar.xz
check_download

cudart_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/${system}-${arch}/cuda_cudart-${system}-${arch}-${cudart_version}-archive.tar.xz
printfln "Downloading CUDART from: ${BLUE}$cudart_url${NC}"
printfln "wget $cudart_url -O ${target_dir}/cuda-cudart-dev-${cudart_version}.tar.xz"
wget "$cudart_url" -O ${target_dir}/cuda-cudart-dev-${cudart_version}.tar.xz
check_download

cupti_url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/${system}-${arch}/cuda_cupti-${system}-${arch}-${cupti_version}-archive.tar.xz
printfln "Downloading CUPTI from: ${BLUE}$cupti_url${NC}"
printfln "wget $cupti_url -O ${target_dir}/cuda-cupti-${cupti_version}.tar.xz"
wget "$cupti_url" -O ${target_dir}/cuda-cupti-${cupti_version}.tar.xz
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

flir_url="https://github.com/FlagTree/flir/archive/refs/heads/main.zip"
printfln "Downloading FLIR from: ${BLUE}$flir_url${NC}"
printfln "wget $flir_url -O ${target_dir}/flir-main.zip"
wget "$flir_url" -O ${target_dir}/flir-main.zip
check_download

triton_shared_url=https://github.com/microsoft/triton-shared/archive/5842469a16b261e45a2c67fbfc308057622b03ee.zip
printfln "Downloading Triton_Shared from: ${BLUE}$triton_shared_url${NC}"
printfln "wget $triton_shared_url -O ${target_dir}/triton-shared-5842469a16b261e45a2c67fbfc308057622b03ee.zip"
wget "$triton_shared_url" -O ${target_dir}/triton-shared-5842469a16b261e45a2c67fbfc308057622b03ee.zip
check_download

printfln " =================== Done ==================="
