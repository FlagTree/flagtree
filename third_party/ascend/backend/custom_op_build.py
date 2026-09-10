# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build raw-address ABI adapters for the CANN 9.0 custom lowering."""
import os
import platform
import shutil
from pathlib import Path


def _find_ccec() -> str:
    env = os.environ.get("CCEC") or os.environ.get("BISHENG")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ccec") or shutil.which("bisheng")
    if found:
        return found
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if toolkit:
        cand = Path(toolkit) / "compiler" / "ccec_compiler" / "bin" / "ccec"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("ccec/bisheng not found; source the CANN set_env script")


def _tikcpp_include() -> Path:
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if toolkit:
        tik = Path(toolkit) / (platform.machine() + "-linux") / "tikcpp" / "tikcfw"
        if (tik / "kernel_operator.h").is_file():
            return tik
    tik = Path("/usr/local/Ascend/ascend-toolkit/latest") / (platform.machine() + "-linux/tikcpp/tikcfw")
    if (tik / "kernel_operator.h").is_file():
        return tik
    raise FileNotFoundError("tikcpp/tikcfw/kernel_operator.h not found")


def _cxx_includes() -> list[str]:
    incs: list[str] = []
    for ver in ("12", "11", "13"):
        base = Path(f"/usr/include/c++/{ver}")
        if (base / "cstdint").is_file():
            incs.extend([f"-I{base}", f"-I/usr/include/{platform.machine()}-linux-gnu/c++/{ver}"])
            break
    return incs


def compile_cmd(src: Path, out: Path) -> list[str]:
    tik = _tikcpp_include()
    return [
        _find_ccec(),
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-aicore-only",
        "-std=c++17",
        f"-I{tik}",
        f"-I{tik / 'interface'}",
        f"-I{tik / 'impl'}",
        *_cxx_includes(),
        # CANN 9.0 rejects ``-emit-llvm`` unless ``-c`` is also present.
        # The object is real LLVM bitcode (magic BC\\xc0\\xde), which
        # ``--link-aicore-bitcode`` can consume. Without ``-emit-llvm``
        # ccec writes a cube ELF relocatable that hivmc cannot link.
        "-emit-llvm",
        "-c",
        str(src),
        "-o",
        str(out),
    ]


def makefile_compile() -> str:
    """Recipe for the CustomOp ``compile`` attribute (``$<`` / ``$@``)."""
    tik = _tikcpp_include()
    incs = " ".join(_cxx_includes())
    return (f"{_find_ccec()} -x cce --cce-aicore-arch=dav-c220-cube --cce-aicore-only "
            f"-std=c++17 -I{tik} -I{tik / 'interface'} -I{tik / 'impl'} {incs} "
            f"-emit-llvm -c $< -o $@")
