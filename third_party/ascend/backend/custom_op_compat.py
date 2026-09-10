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
"""CANN 9.0 lowering for explicitly adapted Ascend custom primitives.

Invoked by the backend compiler stages, never installed as a global hook.
CANN 9.1+ and kernels without adapted custom symbols use the native path.
"""
from __future__ import annotations
import functools
import os
import platform
import re
import shlex
from pathlib import Path

TEXT_IR_PREFIX = b"TRITON_CANN90_CUSTOM_TEXT\n"
SYMBOL_PREFIX = "triton_cann90_"


@functools.lru_cache(None)
def is_cann90():
    root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not root:
        return False
    for filename in ("ascend_toolkit_install.info", "ascend_all_cann_install.info"):
        info = Path(root) / (platform.machine() + "-linux") / filename
        if info.is_file():
            match = re.search(r"^version=(\d+)\.(\d+)(?:\.|$)", info.read_text(), re.M)
            if match:
                return match.groups() == ("9", "0")
    return False


def needs_compat(text):
    return is_cann90() and "hivm.hir.custom" in text and any(
        m.group(1).startswith(SYMBOL_PREFIX) for m in _SYMBOL_RE.finditer(text))


@functools.lru_cache(None)
def cache_directory():
    from triton.runtime.cache import get_cache_manager
    import hashlib
    root = Path(os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME", "."))
    parts = [
        Path(__file__).read_bytes(),
        Path(__file__).with_name("custom_op_build.py").read_bytes(),
        str(root.resolve()).encode(),
        os.environ.get("CCEC", "").encode()
    ]
    for name in ("ascend_toolkit_install.info", "ascend_all_cann_install.info"):
        path = root / (platform.machine() + "-linux") / name
        if path.is_file():
            parts.append(path.read_bytes())
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()
    return Path(get_cache_manager(key).cache_dir)


_SEGMENT3_RE = re.compile(r"operandSegmentSizes\s*=\s*array<i32:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>")
_EMPTY_TMPS_RE = re.compile(r"\s*tmps\(\s*\)")
_SYMBOL_RE = re.compile(r'symbol\s*=\s*"([^"]+)"')


def _split_ins_outs(fragment: str) -> tuple[str, str]:
    """Split ``vals : types`` from an ``ins(...)`` / ``outs(...)`` body."""
    vals, tys = fragment.split(":", 1)
    return vals.strip(), tys.strip()


def _extract_balanced(src: str, open_idx: int) -> tuple[str, int]:
    open_c = src[open_idx]
    close_c = {"(": ")", "{": "}", "[": "]"}[open_c]
    depth = 0
    for j in range(open_idx, len(src)):
        ch = src[j]
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return src[open_idx + 1:j], j + 1
    raise ValueError(f"unbalanced {open_c} in HIVM custom op")


def lower_custom_op_to_call(mlir: str) -> str:
    """Replace ``hivm.hir.custom`` with an i64 ``func.call`` of ``_mlir_ciface_*``.

    CANN 9.0 hivmc does not implement ``hivm.hir.custom``. Passing memref
    descriptors into ``func.call`` fails later: HIVM wraps the AIC body and
    ``llvm.call`` cannot use values defined outside that region. Extract GM
    addresses next to the call (new SSA, new type) so the call stays legal.
    Operations declaring
    ``extra_attr="triton_pass_outputs=true"`` receive output buffer addresses
    after their inputs. This preserves the adapted primitive signatures.
    """
    decls: list[str] = []
    pieces: list[str] = []
    pos = 0
    sink_n = 0
    while True:
        hit = mlir.find("hivm.hir.custom", pos)
        if hit < 0:
            pieces.append(mlir[pos:])
            break
        line_start = mlir.rfind("\n", 0, hit) + 1
        indent_and_assign = mlir[line_start:hit]
        indent_m = re.match(r"^(\s*)", indent_and_assign)
        indent = indent_m.group(1) if indent_m else ""
        pieces.append(mlir[pos:line_start])

        ins_i = mlir.find("ins(", hit)
        if ins_i < 0:
            raise ValueError("hivm.hir.custom is missing ins")
        ins_body, cursor = _extract_balanced(mlir, ins_i + 3)
        # The printer omits outs() for side-effect-only primitives. Never
        # search into the next custom op, which can consume entire regions.
        _outs_body = ""
        next_i = cursor
        while next_i < len(mlir) and mlir[next_i] in " \t":
            next_i += 1
        if mlir.startswith("outs(", next_i):
            _outs_body, cursor = _extract_balanced(mlir, next_i + 4)
        rest_head = mlir[cursor:cursor + 16]
        if rest_head.lstrip().startswith("tmps("):
            tmps_i = mlir.find("tmps(", cursor)
            _tmps_body, cursor = _extract_balanced(mlir, tmps_i + 4)
        attrs_body = ""
        skip = 0
        while cursor + skip < len(mlir) and mlir[cursor + skip] in " \t\n":
            skip += 1
        if cursor + skip < len(mlir) and mlir[cursor + skip] == "{":
            attrs_body, cursor = _extract_balanced(mlir, cursor + skip)
        loc = ""
        loc_i = mlir.find("loc(", cursor)
        newline_i = mlir.find("\n", cursor)
        if loc_i >= 0 and (newline_i < 0 or loc_i < newline_i):
            _loc_body, after_loc = _extract_balanced(mlir, loc_i + 3)
            loc = " " + mlir[loc_i:after_loc]
            cursor = after_loc
        if cursor < len(mlir) and mlir[cursor] == "\n":
            cursor += 1

        symbol_m = _SYMBOL_RE.search(attrs_body) or _SYMBOL_RE.search(mlir[hit:cursor])
        if symbol_m is None:
            raise ValueError("hivm.hir.custom is missing symbol")
        if not symbol_m.group(1).startswith(SYMBOL_PREFIX):
            raise ValueError("CANN 9.0 cannot mix adapted and unadapted custom ABIs in one kernel")
        iface = f"_mlir_ciface_{symbol_m.group(1)}"
        ins_vals, ins_tys = _split_ins_outs(ins_body)
        operands = _split_mlir_list(ins_vals)
        types = _split_mlir_list(ins_tys)
        fragment = mlir[hit:cursor]
        if "triton_pass_outputs=true" in fragment:
            out_vals, out_tys = _split_ins_outs(_outs_body)
            operands.extend(_split_mlir_list(out_vals))
            types.extend(_split_mlir_list(out_tys))
        if len(operands) != len(types):
            raise ValueError("hivm.hir.custom operand/type count mismatch")
        call_vals: list[str] = []
        call_tys: list[str] = []
        prefix: list[str] = []
        for name, ty in zip(operands, types):
            # L0C acc is still a tensor at HIVM input. Materialize a memref so
            # we can extract the on-chip address; hivmc 9.0 cannot pass tensors
            # or memref descriptors through llvm.call.
            if ty.startswith("tensor"):
                m_name = f"%fp_m{sink_n}"
                sink_n += 1
                memref_ty = "memref" + ty[len("tensor"):]
                # CANN 9.0 only accepts ``to_memref %t : memref<...>``.
                prefix.append(f"{indent}{m_name} = bufferization.to_memref {name} : {memref_ty}\n")
                name, ty = m_name, memref_ty
            if ty.startswith("memref"):
                p_name = f"%fp_p{sink_n}"
                i_name = f"%fp_i{sink_n}"
                sink_n += 1
                prefix.append(f"{indent}{p_name} = memref.extract_aligned_pointer_as_index "
                              f"{name} : {ty} -> index\n")
                prefix.append(f"{indent}{i_name} = arith.index_cast {p_name} : index to i64\n")
                call_vals.append(i_name)
                call_tys.append("i64")
            else:
                call_vals.append(name)
                call_tys.append(ty)
        pieces.extend(prefix)
        pieces.append(f"{indent}func.call @{iface}({', '.join(call_vals)}) "
                      f": ({', '.join(call_tys)}) -> (){loc}\n")
        # hivmc on CANN 9.0 rejects hivm.vf_mode / hivm.pipe on func.func.
        # i64 callees need an explicit matching core type for hivmc.
        vector_call = "#hivm.tcore_type<VECTOR>" in fragment
        func_core = "AIV" if vector_call else "AIC"
        tensor_core = "VECTOR" if vector_call else "CUBE"
        attrs = [
            f"hivm.func_core_type = #hivm.func_core_type<{func_core}>",
            "hivm.part_of_mix",
            f"hivm.tcore_type = #hivm.tcore_type<{tensor_core}>",
        ]
        decl = (f"  func.func private @{iface}({', '.join(call_tys)}) "
                f"attributes {{{', '.join(attrs)}}}")
        if decl not in decls:
            decls.append(decl)
        pos = cursor

    lowered = "".join(pieces)
    if decls:
        end = lowered.rfind("}")
        if end < 0:
            raise ValueError("cannot insert CustomOp callee: no module end")
        lowered = lowered[:end] + "\n".join(decls) + "\n" + lowered[end:]
    return lowered


def rewrite_custom_op_segments(mlir: str) -> str:
    """Flatten 3-element CustomOp segment sizes to CANN 9.0's 2-element form."""

    def _repl(match: re.Match[str]) -> str:
        ins, outs, tmps = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if tmps != 0:
            raise ValueError(f"hivm.hir.custom has non-empty tmps={tmps}; cannot lower to CANN 9.0")
        return f"operandSegmentSizes = array<i32: {ins}, {outs}>"

    return _EMPTY_TMPS_RE.sub("", _SEGMENT3_RE.sub(_repl, mlir))


def _split_mlir_list(src: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in src:
        if ch in "(<":
            depth += 1
            buf.append(ch)
        elif ch in ")>":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf).strip())
    return [x for x in items if x]


def _skip_mlir_type(src: str, i: int) -> int:
    while i < len(src) and src[i] in " \t":
        i += 1
    while i < len(src) and (src[i].isalnum() or src[i] in "._!"):
        i += 1
    while i < len(src) and src[i] == "<":
        depth = 0
        while i < len(src):
            if src[i] == "<":
                depth += 1
            elif src[i] == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    return i


def rewrite_cann90_bufferization(mlir: str) -> str:
    """Drop FlagTree's ``to_tensor %x : memref<T> to tensor<T>`` result type.

    CANN 9.0 hivmc only parses ``bufferization.to_tensor %x : memref<T>``.
    """
    key = "bufferization.to_tensor"
    out: list[str] = []
    pos = 0
    while True:
        hit = mlir.find(key, pos)
        if hit < 0:
            out.append(mlir[pos:])
            break
        colon = mlir.find(":", hit)
        newline = mlir.find("\n", hit)
        if colon < 0 or (newline >= 0 and colon > newline):
            out.append(mlir[pos:hit + len(key)])
            pos = hit + len(key)
            continue
        ty_end = _skip_mlir_type(mlir, colon + 1)
        rest = mlir[ty_end:ty_end + 16]
        if rest.lstrip().startswith("to "):
            to_i = mlir.find("to ", ty_end)
            ty_end = _skip_mlir_type(mlir, to_i + 2)
            out.append(mlir[pos:colon])
            out.append(mlir[colon:mlir.find("to ", colon)])
            pos = ty_end
        else:
            out.append(mlir[pos:ty_end])
            pos = ty_end
    return "".join(out)


def prepare_linalg(text):
    """Keep custom operations until core assignment, but use CANN 9.0 syntax."""
    # The actual custom core annotations determine whether this is a mixed
    # kernel; do not allow a stale caller-provided AIC-only mode to drop AIV.
    if "#hivm.tcore_type<CUBE>" in text and "#hivm.tcore_type<VECTOR>" in text:
        text = re.sub(r'mix_mode\s*=\s*"[^"]+"', 'mix_mode = "mix"', text)
    return rewrite_custom_op_segments(rewrite_cann90_bufferization(text))


def prepare_hivmc_mlir(text):
    return lower_custom_op_to_call(prepare_linalg(text))


def compiler_command(text, compiler, env):
    """Wrap only this compiler subprocess so its hivmc reads the adapted ABI."""
    if not needs_compat(text):
        return compiler, env
    root = Path(os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"])
    # Do not nest a wrapper left on PATH by an application importing FlagGems.
    candidates = (root / "bin/bishengir-compile", root / "tools/bishengir/bin/bishengir-compile")
    real = next((p.resolve() for p in candidates if p.is_file()), None)
    if real is None or not real.with_name("hivmc").is_file():
        raise RuntimeError("CANN 9.0 bishengir-compile and sibling hivmc were not found")
    import hashlib
    directory = cache_directory() / hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "hivmc"
    shim.write_text("""#!/usr/bin/env python3
import importlib.util
import os
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("custom_op_compat", os.environ["TRITON_CUSTOM_COMPAT_MODULE"])
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)
for arg in sys.argv[1:]:
    path = Path(arg)
    if path.suffix == ".mlir" and path.is_file():
        text = path.read_text()
        if compat.needs_compat(text):
            path.write_text(compat.prepare_hivmc_mlir(text))
real = os.environ["TRITON_CUSTOM_REAL_HIVMC"]
os.execv(real, [real, *sys.argv[1:]])
""")
    shim.chmod(0o755)
    wrapper = directory / "bishengir-compile"
    wrapper.write_text('#!/bin/bash\nset -euo pipefail\nexec -a "$0" ' + shlex.quote(str(real)) + ' "$@"\n')
    wrapper.chmod(0o755)
    env = dict(env)
    env["PATH"] = str(directory) + os.pathsep + env.get("PATH", "")
    env["TRITON_CUSTOM_COMPAT_MODULE"] = str(Path(__file__).resolve())
    env["TRITON_CUSTOM_REAL_HIVMC"] = str(real.with_name("hivmc"))
    return str(wrapper), env
