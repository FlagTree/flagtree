#!/bin/bash

PROXY=http://192.168.100.225:8889
setup_proxy() {
    # Downloading python requirement is needed.
    export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
    export HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY ALL_PROXY=$PROXY
}

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

use_venv=OFF
if [ $# -gt 0 ]; then
    if [[ "${1,,}" == "venv" ]]; then
        use_venv=ON
    fi
fi

if [ "x$use_venv" == "xON" ]; then
    python3 -m venv $project_dir/.venv --prompt flagtree
    source $project_dir/.venv/bin/activate
fi

setup_proxy

apt install git
apt install lld

pip3 install -r $project_dir/third_party/tsingmicro/requirements.txt

pip3 install -r $project_dir/python/requirements.txt
