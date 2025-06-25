[English](./README.md)

## FlagTree

FlagTree 是面向多种 AI 芯片的开源、统一编译器。FlagTree 致力于打造多元 AI 芯片编译器及相关工具平台，发展和壮大 Triton 上下游生态。项目当前处于初期，目标是兼容现有适配方案，统一代码仓库，快速实现单仓库多后端支持。对于上游模型用户，提供多后端的统一编译能力；对于下游芯片厂商，提供 Triton 生态接入范例。

## 从源代码安装
安装依赖（注意使用正确的 python3.x 执行）：
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

编译安装，目前支持的后端 backendxxx 包括 iluvatar、xpu、mthreads、cambricon（有限支持）：
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
```

## 构建技巧

自动下载依赖库的速度可能受限于网络环境，编译前可自行下载至缓存目录 ~/.flagtree（可通过环境变量 FLAGTREE_CACHE_DIR 修改），无需自行设置 LLVM_BUILD_DIR 等环境变量。 <br>
各后端完整编译命令如下： <br>

[iluvatar](/third_party/iluvatar/)
```shell
# 推荐使用镜像 Ubuntu 20.04
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
# 推荐使用镜像（22GB）https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
# 联系 kunlunxin-support@baidu.com 可获取进一步支持
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
# 推荐使用镜像 flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz
tar zxvf mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```
[aipu (arm npu)](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/)
```shell
# 推荐使用镜像 Ubuntu 20.04
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
# 推荐使用镜像 Ubuntu 20.04
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://github.com/FlagTree/flagtree/releases/download/v0.2.0-build-deps/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
python3 -m pip install . --no-build-isolation -v
```
[ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/python/setup_tools/setup_helper.py)
```shell
# 推荐使用镜像 flagtree/dockerfiles/Dockerfile-ubuntu20.04-python3.9-ascend
# 在 https://www.hiascend.com/developer/download/community/result?module=cann
# 注册账号后下载对应平台的 cann-toolkit、cann-kernels
# 这里以 AArch64 架构的 A3 处理器为例展示如何安装
chmod +x Ascend-cann-toolkit_8.2.RC1.alpha002_linux-aarch64.run
./Ascend-cann-toolkit_8.2.RC1.alpha002_linux-aarch64.run --install
chmod +x Atlas-A3-cann-kernels_8.1.RC1_linux-aarch64.run
./Atlas-A3-cann-kernels_8.1.RC1_linux-aarch64.run --install
# 编译安装
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-b5cc222d-ubuntu-arm64.tar.gz
tar zxvf llvm-b5cc222d-ubuntu-arm64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
python3 -m pip install . --no-build-isolation -v
```

使用默认的编译命令，可以编译安装 nvidia、amd、triton_shared (cpu) 后端：
```shell
# 自行下载 llvm
cd ${YOUR_LLVM_DOWNLOAD_DIR}
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
# 编译安装
cd ${YOUR_CODE_DIR}/flagtree/python
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# 如果接下来需要编译安装其他后端，应清空 LLVM 相关环境变量
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

## 运行测试

安装完成后可以在设备支持的环境下运行测试：
```shell
# nvidia
cd python/test
python3 -m pytest -s
# other backends
cd third_party/backendxxx/python/test
python3 -m pytest -s
python3 test_xxx.py
```

## 关于贡献

欢迎参与 FlagTree 的开发并贡献代码，详情请参考[CONTRIBUTING.md](/CONTRIBUTING_cn.md)。

## 许可证

FlagTree 使用 [MIT license](/LICENSE)。
