#!/bin/bash


echo " =================== Start Downloading Offline Build Files ==================="
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

# detect system arch
arch=$(uname -m)
echo "Detected the system arch: $arch"
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

check_download() {
    if [ $? -eq 0 ]; then
        echo "Download Success"
    else
        echo "Download Failed !!!"
        exit 1
    fi
    echo ""
}

if [ $# -ge 1 ]; then
    target_dir="$1"
else
    target_dir="$HOME/.flagtree-offline-download"
fi

echo ""
if [ ! -d "$target_dir" ]; then
    echo "Creating download output directory $target_dir"
    mkdir -p "$target_dir"
else
    echo "Download output directory $target_dir already exists"
fi
echo ""

nvcc_url=https://anaconda.org/nvidia/cuda-nvcc/${nv_toolchain_version}/download/linux-${arch}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2
echo "Downloading NVCC from: $nvcc_url"
echo "wget $nvcc_url -O ${target_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
wget "$nvcc_url" -O ${target_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2
check_download

cuobjdump_url=https://anaconda.org/nvidia/cuda-cuobjdump/${nv_toolchain_version}/download/linux-${arch}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2
echo "Downloading CUOBJBDUMP from: $cuobjdump_url"
echo "wget $cuobjdump_url -O ${target_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2"
wget "$cuobjdump_url" -O ${target_dir}/cuda-cuobjdump-${nv_toolchain_version}-0.tar.bz2
check_download

nvdisam_url=https://anaconda.org/nvidia/cuda-nvdisasm/${nv_toolchain_version}/download/linux-${arch}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2
echo "Downloading NVDISAM from: $nvdisam_url"
echo "wget $nvdisam_url -O ${target_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2"
wget "$nvdisam_url" -O ${target_dir}/cuda-nvdisasm-${nv_toolchain_version}-0.tar.bz2
check_download

cudart_url=https://anaconda.org/nvidia/cuda-cudart-dev/${nv_toolchain_version}/download/linux-${arch}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2
echo "Downloading CUDART from: $cudart_url"
echo "wget $cudart_url -O ${target_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2"
wget "$cudart_url" -O ${target_dir}/cuda-cudart-dev-${nv_toolchain_version}-0.tar.bz2
check_download

cupti_url=https://anaconda.org/nvidia/cuda-cupti/${nv_toolchain_version}/download/linux-${arch}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2
echo "Downloading CUPTI from: $cupti_url"
echo "wget $cupti_url -O ${target_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2"
wget "$cupti_url" -O ${target_dir}/cuda-cupti-${nv_toolchain_version}-0.tar.bz2
check_download

pybind11_url=https://github.com/pybind/pybind11/archive/refs/tags/v${pybind11_version}.tar.gz
echo "Downloading Pybind11 from: $pybind11_url"
echo "wget $pybind11_url -O ${target_dir}/pybind11-${pybind11_version}.tar.gz"
wget "$pybind11_url" -O ${target_dir}/pybind11-${pybind11_version}.tar.gz
check_download

json_url=https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
echo "Downloading JSON library from: $json_url"
echo "wget $json_url -O ${target_dir}/include.zip"
wget "$json_url" -O ${target_dir}/include.zip
check_download

googletest_url=https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
echo "Downloading GoogleTest from: $googletest_url"
echo "wget $googletest_url -O ${target_dir}/googletest-release-1.12.1.zip"
wget "$googletest_url" -O ${target_dir}/googletest-release-1.12.1.zip
check_download

triton_shared_url=https://github.com/microsoft/triton-shared/archive/380b87122c88af131530903a702d5318ec59bb33.zip
echo "Downloading Triton_Shared from: $triton_shared_url"
echo "wget $triton_shared_url -O ${target_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip"
wget "$triton_shared_url" -O ${target_dir}/triton-shared-380b87122c88af131530903a702d5318ec59bb33.zip
check_download

echo " =================== Done ==================="