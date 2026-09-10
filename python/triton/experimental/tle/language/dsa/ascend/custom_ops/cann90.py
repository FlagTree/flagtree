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
"""CANN 9.0 ABI configuration for supported public Ascend primitives."""
import functools
import hashlib
import subprocess
from pathlib import Path
from triton.backends.ascend import custom_op_build as build_utils
from triton.backends.ascend.custom_op_compat import cache_directory, is_cann90

CUSTOM = Path(__file__).resolve().parent


def template_include():
    import os

    override = os.environ.get("FLAGTREE_TEMPLATE_INCLUDE")
    if override and (Path(override) / "Utils.h").is_file():
        return Path(override)
    for root in CUSTOM.parents:
        candidate = (root / "third_party/ascend/AscendNPU-IR/bishengir/lib/Template/include")
        if (candidate / "Utils.h").is_file():
            return candidate
    raise RuntimeError("CANN 9.0 custom ABI requires FlagTree Template headers; "
                       "set FLAGTREE_TEMPLATE_INCLUDE to the directory containing Utils.h")


SOURCES = {
    "compare_scalar": ("mask_ops/compare_scalar.cpp", "custom_compare_scalar_float"),
    "gather_mask": ("mask_ops/gather_mask.cpp", "custom_gather_mask_float"),
    "cast_int4_to_fp16": ("cast_ops/cast_int4_to_fp16.cpp", "custom_cast_int4_to_fp16"),
}


@functools.lru_cache(None)
def build(kind, n):
    BUILD = cache_directory() / "adapters"
    INCLUDE = template_include()
    relative, callee = SOURCES[kind]
    source = CUSTOM / relative
    if kind == "compare_scalar":
        signature = "int64_t src, float scalar, int64_t dst"
        body = (f"auto s = view<float>(src, {n}); auto d = view<uint16_t>(dst, {n // 16});\n    "
                f"_mlir_ciface_{callee}(&s, scalar, &d);")
    elif kind == "gather_mask":
        signature = "int64_t src, int64_t mask, int64_t dst, int64_t count"
        body = (f"auto s = view<float>(src, {n}); auto m = view<uint16_t>(mask, {n // 16});\n    "
                f"auto d = view<float>(dst, {n}); auto c = view<int32_t>(count, 8);\n    "
                f"_mlir_ciface_{callee}(&s, &m, &d, &c);")
    else:
        signature = "int64_t src, int64_t dst"
        body = (f"auto s = view<uint8_t>(src, {n}); auto d = view<half>(dst, {2 * n});\n    "
                f"_mlir_ciface_{callee}(&s, &d);")
    code = f"""#include "{source}"
template <typename T>
[aicore] __attribute__((always_inline)) memref_t<__ubuf__ T, 1> view(int64_t address, int64_t size) {{
    auto pointer = reinterpret_cast<__ubuf__ T *>(address);
    return {{pointer, pointer, 0, {{size}}, {{1}}}};
}}
extern "C" [aicore] __attribute__((always_inline)) void
_mlir_ciface_ABI_ENTRY({signature}) {{
    {body}
}}
"""
    digest = hashlib.sha256(
        (code + source.read_text() + (CUSTOM / "mask_ops/mask_common.h").read_text()).encode()).hexdigest()[:16]
    symbol = f"triton_cann90_{kind}_{n}_{digest}"
    BUILD.mkdir(parents=True, exist_ok=True)
    cpp, bc = BUILD / (symbol + ".cpp"), BUILD / (symbol + ".bc")
    cpp.write_text(code.replace("ABI_ENTRY", symbol))
    if not bc.exists():
        command = [x.replace("dav-c220-cube", "dav-c220-vec") for x in build_utils.compile_cmd(cpp, bc)]
        subprocess.run(command + ["-O3", "-I" + str(INCLUDE)], check=True)
    return symbol, cpp, bc


def _configure_vector(instance, kind, src):
    symbol, cpp, bc = build(kind, src.numel.value)
    instance.symbol = symbol
    instance.source, instance.bitcode = str(cpp), str(bc)
    instance.extra_attr = "triton_pass_outputs=true"
    instance.compile = (build_utils.makefile_compile().replace("dav-c220-cube", "dav-c220-vec") + " -O3 -I" +
                        str(template_include()))


def configure(instance, kind, src=None):
    if not is_cann90():
        return
    if kind not in ("cube_begin", "cube_end"):
        _configure_vector(instance, kind, src)
        return
    source = CUSTOM / "sync_ops/cube_boundary.cpp"
    directory = cache_directory() / "adapters"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    symbol = "triton_cann90_" + kind
    cpp = directory / ("cube_boundary_" + digest + ".cpp")
    bc = cpp.with_suffix(".bc")
    cpp.write_text(source.read_text().replace("custom_cube_begin",
                                              "triton_cann90_cube_begin").replace("custom_cube_end",
                                                                                  "triton_cann90_cube_end"))
    if not bc.exists():
        subprocess.run(build_utils.compile_cmd(cpp, bc) + ["-O3"], check=True)
    instance.symbol, instance.source, instance.bitcode = symbol, str(cpp), str(bc)
    instance.compile = build_utils.makefile_compile() + " -O3"
