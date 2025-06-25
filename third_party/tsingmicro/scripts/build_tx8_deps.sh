#!/bin/bash

# # 定义一个函数，用于克隆 Git 仓库并切换到指定分支或指定 commit
clone_and_checkout() {
    local git_url="$1"
    local target_dir="$2"
    local ref_type="$3"  # "branch" 或 "commit"
    local ref_value="$4" # 分支名称或 commit ID

    # 检查目标目录是否存在，如果不存在则创建
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create target directory: $target_dir"
            return 1
        fi
    fi

    if [ "$(ls -A $target_dir)" ]; then
        echo "jump clone $target_dir"
        return 1
    fi

    # 使用 pushd 进入目标目录
    pushd "$target_dir" >/dev/null || return 1

    # 克隆仓库
    git clone "$git_url" .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone repository: $git_url"
        popd >/dev/null
        return 1
    fi

    # 根据 ref_type 切换到分支或 commit
    if [ "$ref_type" == "branch" ]; then
        # 检查分支是否存在
        if ! git branch -r | grep -q "origin/$ref_value"; then
            echo "Error: Branch '$ref_value' does not exist in the repository."
            popd >/dev/null
            return 1
        fi
        # 切换到指定分支
        git checkout "$ref_value"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to switch to branch: $ref_value"
            popd >/dev/null
            return 1
        fi
        echo "Successfully cloned and switched to branch '$ref_value' for repository: $git_url"
    elif [ "$ref_type" == "commit" ]; then
        # 检查 commit 是否存在
        if ! git rev-parse "$ref_value" >/dev/null 2>&1; then
            echo "Error: Commit '$ref_value' does not exist in the repository."
            popd >/dev/null
            return 1
        fi
        # 切换到指定 commit
        git checkout "$ref_value"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to switch to commit: $ref_value"
            popd >/dev/null
            return 1
        fi
        echo "Successfully cloned and switched to commit '$ref_value' for repository: $git_url"
    else
        echo "Error: Invalid ref_type. Use 'branch' or 'commit'."
        popd >/dev/null
        return 1
    fi

    # 使用 popd 退出目录
    popd >/dev/null
    return 0
}

download_and_extract() {
    local url="$1"       # 下载链接
    local target_dir="$2" # 目标目录
    local temp_dir="$3"   # 临时目录
    local tag_name="$4" # 标签

    temp_dir=$temp_dir/$tag_name

    # 确保目标目录和临时目录存在
    mkdir -p "$target_dir"
    mkdir -p "$temp_dir"

    # 检查目标目录是否为空
    if [ -z "$(ls -A "$target_dir")" ]; then
        echo "目标目录 $target_dir 为空，开始下载并解压..."

        # 下载文件到临时目录
        local temp_file="$temp_dir/$(basename "$url")"

        if [[ ! -f $temp_file ]]; then
            wget -O "$temp_file" "$url"
        else
            echo "文件 $temp_file 已存在，跳过下载."
        fi

        # 检查下载是否成功
        if [ $? -eq 0 ]; then
            # 解压到临时目录
            unzip_dir=$temp_dir/$(date +"%Y_%m_%d")
            mkdir -p $unzip_dir
            tar -xz -C "$unzip_dir" -f "$temp_file"
            echo "解压到:$unzip_dir"

            cp -r $unzip_dir/* $target_dir

            # 检查解压后的内容
            # local extracted_dir
            # extracted_dir=$(ls -d "$temp_dir"/*/ | head -n 1)
            # if [ -d "$extracted_dir" ]; then
            #     if [ -d "$target_dir" ]; then
            #         rm -rf $target_dir
            #     fi
            #     # 移动解压后的目录到目标目录
            #     mv "$extracted_dir" "$target_dir"
            #     echo "下载并解压完成，$extracted_dir 目录已移动到 $target_dir。"
            # else
            #     echo "解压后没有找到目录，无法移动。"
            #     exit 1
            # fi
        else
            echo "下载失败，退出脚本。"
            exit 1
        fi
    else
        echo "目标目录 $target_dir 不为空，跳过下载操作。"
    fi
}

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

build_tx8_deps=OFF

if [ $# -gt 0 ]; then
    if [[ "${1,,}" == "build" ]]; then
        build_tx8_deps=ON
    fi
fi

if [ "x$build_tx8_deps" == "xON" ]; then
    download_dir=$WORKSPACE/download
    tx8fw_dir=$download_dir/triton-tx8fw
    # download_and_extract "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_2025-0606_bbe682.tar.gz" \
    #     "$tx8fw_dir" "$WORKSPACE/download" "tx8fw"
    download_and_extract "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_2025-0617_acd719.tar.gz" \
        "$tx8fw_dir" "$download_dir" "tx8fw"

    host_runtime_dir=$download_dir/host_runtime
    download_and_extract "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx8-host/master/host_runtime_v5.2.0_daily_2025-0605_7a6768.tar.gz" \
        "$host_runtime_dir" "$download_dir" "runtime"

    xuantie_sdk_dir=$download_dir/tx8fw-xuantie-sdk
    clone_and_checkout "git@gitlab.tsingmicro.com:tx8_developers/tx8fw-xuantie-sdk.git" \
        "$xuantie_sdk_dir" "branch" "master"

    kcore_fw_bin=$tx8fw_dir/bin/FW/kcore_fw.bin
    if [ ! -f $kcore_fw_bin ]; then
        echo "error can't find:$kcore_fw_bin"
    fi
    instr_tx81_lib=$tx8fw_dir/lib/libinstr_tx81.a
    if [ ! -f $instr_tx81_lib ]; then
        echo "error can't find:$instr_tx81_lib"
    fi
    instr_tx81_inc=$tx8fw_dir/include/instr_tx81/include
    if [ ! -d $instr_tx81_inc ]; then
        echo "error can't find:$instr_tx81_inc"
    fi
    # instr_tx81_lib=$WORKSPACE/download/libinstr_tx81.a
    xuantie_dir=$xuantie_sdk_dir/Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2
    if [ ! -d $xuantie_dir ]; then
        echo "error can't find:$xuantie_dir"
    fi

    tx8_depends_dir=$WORKSPACE/tx8_deps
    if [ -d $tx8_depends_dir ]; then
        rm -rf $tx8_depends_dir
    fi
    mkdir $tx8_depends_dir
    pushd $tx8_depends_dir
        cp -r $xuantie_dir ./
        cp -r $host_runtime_dir/**/* ./

        if [ ! -d chip_out ]; then
            mkdir lib
        fi
        cp $kcore_fw_bin chip_out

        if [ ! -d lib ]; then
            mkdir lib
        fi
        cp $instr_tx81_lib lib
        cp $instr_tx81_inc/* include

        # 非必须
        lib_log_h=$tx8fw_dir/include/components/oplib_tx81/riscv/riscv/include/lib_log.h
        echo "lib_log_h:$lib_log_h"
        if [ -f $lib_log_h ]; then
            cp $lib_log_h ./include
        fi
    popd

    pushd $WORKSPACE
        current_time=$(date +%Y%m%d_%H%M%S)
        pkg_file=download/tx8_depends_$current_time.tar.gz
        if [ ! -d download ]; then
            mkdir download
        fi
        if [ -f $pkg_file ]; then
            rm -f $pkg_file
        fi
        tar -zcvf $pkg_file tx8_deps
    popd
else
    echo abc
    # tx8_deps_base=$WORKSPACE/tx8_deps
    # # clone_and_checkout "git@gitlab.tsingmicro.com:triton-based-projects/llvm-project.git" "$WORKSPACE/llvm-project-for-ztc" "branch" "ztc"
    # clone_and_checkout "git@gitlab.tsingmicro.com:triton-based-projects/llvm-project.git" "$WORKSPACE/llvm-project" "commit" "a66376b0dc3b2ea8a84fda26faca287980986f78"

    # download_and_extract "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx8/triton/tx8_depends_20250512_145415.tar.gz" \
    #         "$tx8_deps_base" "$WORKSPACE/download"
    # clone_and_checkout "ssh://192.168.100.107:29418/tx8_toolchain/tx8be-oplib" "third_party/tx8be-oplib" "commit" "b5651a734f1a6a8943765c83bee1e80d6a2c6a37"
fi
