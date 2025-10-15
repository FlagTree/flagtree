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

printfln " =================== Start Packing Downloaded Offline Build Files ==================="
printfln ""
SCRIPT_DIR=$(dirname $0)
# detect nvidia toolchain version requirement
NV_TOOLCHAIN_VERSION_FILE="$SCRIPT_DIR/../../cmake/nvidia-toolchain-version.json"
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
JSON_VERSION_FILE="$SCRIPT_DIR/../../cmake/json-version.txt"
if [ -f "$JSON_VERSION_FILE" ]; then
    json_version=$(tr -d '\n' < "$JSON_VERSION_FILE")
    printfln "JSON Version Required: $json_version"
else
    printfln "${RED}Error: version file $JSON_VERSION_FILE is not exist${NC}"
fi

output_zip="offline-build-pack-triton-3.2.x.zip"

# handle input
printfln ""
if [ $# -ge 1 ]; then
    input_dir="$1"
    printfln "${BLUE}Use $input_dir as input directory${NC}"
else
    printfln "${RED}Error: No input directory specified${NC}"
    printfln "${GREEN}Usage: sh utils/offline_build_pack.sh [input_dir] [output_zip_file]${NC}"
    exit 1
fi

# handle output
if [ $# -ge 2 ]; then
    output_zip="$2"
    printfln "${BLUE}Use $output_zip as output .zip file${NC}"
else
    printfln "${YELLOW}Use default output .zip file name: $output_zip${NC}"
fi

if [ ! -d "$input_dir" ]; then
    printfln "${RED}Error: Cannot find input directory $input_dir${NC}"
    exit 1
else
    printfln "Find input directory: $input_dir"
fi
printfln ""

ptxas_file="cuda-ptxas-${ptxas_version}-0.tar.bz2"
cudacrt_file="cuda-crt-${cudacrt_version}-0.tar.bz2"
cuobjdump_file="cuda-cuobjdump-${cuobjdump_version}-0.tar.bz2"
nvdisasm_file="cuda-nvdisasm-${nvdisasm_version}-0.tar.bz2"
cudart_file="cuda-cudart-dev-${cudart_version}-0.tar.bz2"
cupti_file="cuda-cupti-${cupti_version}-0.tar.bz2"
json_file="include.zip"
googletest_file="googletest-release-1.12.1.zip"
triton_ascend_file="triton-ascend-master.zip"
ascendnpu_ir_file="ascendnpu-ir-1922371c42749fda534d6395b7ed828b5c9f36d4.zip"
triton_file="triton-9641643da6c52000c807b5eeed05edaec4402a67.zip"
triton_shared_file="triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"

if [ ! -f "$input_dir/$ptxas_file" ]; then
    printfln "${RED}Error: File $input_dir/$ptxas_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$ptxas_file"

if [ ! -f "$input_dir/$cudacrt_file" ]; then
    printfln "${RED}Error: File $input_dir/$cudacrt_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$cudacrt_file"

if [ ! -f "$input_dir/$cuobjdump_file" ]; then
    printfln "${RED}Error: File $input_dir/$cuobjdump_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$cuobjdump_file"

if [ ! -f "$input_dir/$nvdisasm_file" ]; then
    printfln "${RED}Error: File $input_dir/$nvdisasm_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$nvdisasm_file"

if [ ! -f "$input_dir/$cudart_file" ]; then
    printfln "${RED}Error: File $input_dir/$cudart_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$cudart_file"

if [ ! -f "$input_dir/$cupti_file" ]; then
    printfln "${RED}Error: File $input_dir/$cupti_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$cupti_file"

if [ ! -f "$input_dir/$json_file" ]; then
    printfln "${RED}Error: File $input_dir/$json_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$json_file"

if [ ! -f "$input_dir/$googletest_file" ]; then
    printfln "${RED}Error: File $input_dir/$googletest_file does not exist, run README_offline_build.sh for more information${NC}"
    exit 1
fi
printfln "Find $input_dir/$googletest_file"

if [ ! -f "$input_dir/$triton_ascend_file" ]; then
    printfln "${YELLOW}Warning: File $input_dir/$triton_ascend_file does not exist. This file is necessary for ascend backend, please check if you need it.${NC}"
    triton_ascend_file=""
else
    printfln "Find $input_dir/$triton_ascend_file"
fi

if [ ! -f "$input_dir/$ascendnpu_ir_file" ]; then
    printfln "${YELLOW}Warning: File $input_dir/$ascendnpu_ir_file does not exist. This file is necessary for ascend backend, please check if you need it.${NC}"
    ascendnpu_ir_file=""
else
    printfln "Find $input_dir/$ascendnpu_ir_file"
fi

if [ ! -f "$input_dir/$triton_file" ]; then
    printfln "${YELLOW}Warning: File $input_dir/$triton_file does not exist. This file is necessary for ascend backend, please check if you need it.${NC}"
    triton_file=""
else
    printfln "Find $input_dir/$triton_file"
fi

if [ ! -f "$input_dir/$triton_shared_file" ]; then
    printfln "${YELLOW}Warning: File $input_dir/$triton_shared_file does not exist. This file is optional, please check if you need it.${NC}"
    triton_shared_file=""
else
    printfln "Find $input_dir/$triton_shared_file"
fi

printfln "cd ${input_dir}"
cd "$input_dir"

printfln "Compressing..."
zip "$output_zip" "$ptxas_file" "$cudacrt_file" "$cuobjdump_file" "$nvdisasm_file" "$cudart_file" "$cupti_file" \
    "$json_file" "$googletest_file" "$triton_ascend_file" "$ascendnpu_ir_file" "$triton_file" "$triton_shared_file"

printfln "cd -"
cd -

printfln ""
if [ $? -eq 0 ]; then
    printfln "${GREEN}Offline Build dependencies are successfully compressed into $output_zip${NC}"
    exit 0
else
    printfln "${RED}Error: Failed to compress offline build dependencies${NC}"
    exit 1
fi
