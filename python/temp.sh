#!/bin/bash

echo -e "Downloading NVCC cudacrt from: ${BLUE}$cudacrt_url${NC}"
echo -e "wget $cudacrt_url -O ${target_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2"
wget "$cudacrt_url" -O ${target_dir}/cuda-nvcc-${nv_toolchain_version}-0.tar.bz2
check_download