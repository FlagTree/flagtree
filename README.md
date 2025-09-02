[中文版](./README_cn.md)

## FlagTree

FlagTree is an open source, unified compiler for multiple AI chips project dedicated to developing a diverse ecosystem of AI chip compilers and related tooling platforms, thereby fostering and strengthening the upstream and downstream Triton ecosystem. Currently in its initial phase, the project aims to maintain compatibility with existing adaptation solutions while unifying the codebase to rapidly implement single-repository multi-backend support. For upstream model users, it provides unified compilation capabilities across multiple backends; for downstream chip manufacturers, it offers examples of Triton ecosystem integration.

## Install from source
Installation dependencies (ensure you use the correct python3.x version):
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

Compile and install. Currently supported backends (backendxxx) include iluvatar, xpu, mthreads, and cambricon (limited support):
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
```

## Tips for building

Automatic dependency library downloads may be limited by network conditions. You can manually download to the cache directory ~/.flagtree (modifiable via the FLAGTREE_CACHE_DIR environment variable). No need to manually set LLVM environment variables such as LLVM_BUILD_DIR.
Complete build commands for each backend:

[iluvatar](/third_party/iluvatar/)
```shell
# Recommended: Use Ubuntu 20.04
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/iluvatar-llvm18-x86_64.tar.gz
tar zxvf iluvatar-llvm18-x86_64.tar.gz
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64.tar.gz
tar zxvf iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```
[xpu (klx)](/third_party/xpu/)
```shell
# Recommended: Use the Docker image (22GB) https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
# Contact kunlunxin-support@baidu.com for support
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/XTDK-llvm19-ubuntu2004_x86_64.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64.tar.gz
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/xre-Linux-x86_64.tar.gz
tar zxvf xre-Linux-x86_64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```
[mthreads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads/)
```shell
# Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz
tar zxvf mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```
[aipu (arm npu)](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/)
```shell
# Recommended: Use Ubuntu 20.04
mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=aipu
python3 -m pip install . --no-build-isolation -v
```
[tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)
```shell
# Recommended: Use Ubuntu 20.04
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://github.com/FlagTree/flagtree/releases/download/v0.2.0-build-deps/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-x64.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
python3 -m pip install . --no-build-isolation -v
```
[ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/python/setup_tools/setup_helper.py)
```shell
# Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu20.04-python3.9-ascend
# After registering an account at https://www.hiascend.com/developer/download/community/result?module=cann,
# download the cann-toolkit and cann-kernels for the corresponding platform.
# Here we use the A3 processor with AArch64 architecture as an example to demonstrate how to install.
chmod +x Ascend-cann-toolkit_8.2.RC1.alpha002_linux-aarch64.run
./Ascend-cann-toolkit_8.2.RC1.alpha002_linux-aarch64.run --install
chmod +x Atlas-A3-cann-kernels_8.1.RC1_linux-aarch64.run
./Atlas-A3-cann-kernels_8.1.RC1_linux-aarch64.run --install
# build
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-b5cc222d-ubuntu-arm64.tar.gz
tar zxvf llvm-b5cc222d-ubuntu-arm64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
python3 -m pip install . --no-build-isolation -v
```

To build with default backends nvidia, amd, triton_shared (cpu):
```shell
# manually download LLVM
cd ${YOUR_LLVM_DOWNLOAD_DIR}
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
tar zxvf llvm-86b69c31-ubuntu-x64.tar.gz
# build
cd ${YOUR_CODE_DIR}/flagtree/python
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-86b69c31-ubuntu-x64
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# If you need to build other backends afterward, you should clear LLVM-related environment variables
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

## Running tests

After installation, you can run tests in the backend directory:
```shell
cd third_party/backendxxx/python/test
python3 -m pytest -s
```

## Contributing

Contributions to FlagTree development are welcome. Please refer to [CONTRIBUTING.md](/CONTRIBUTING_cn.md) for details.

## License

FlagTree is licensed under the [MIT license](/LICENSE).
