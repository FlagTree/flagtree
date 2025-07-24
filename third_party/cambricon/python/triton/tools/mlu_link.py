from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
from argparse import ArgumentParser, RawDescriptionHelpFormatter


def _exists(x):
    return x is not None


class LinkerError(Exception):
    pass


@dataclass
class KernelLinkerMeta:
    orig_kernel_name: str
    arg_names: Sequence[str]
    arg_ctypes: Sequence[str]
    """ number of specialized arguments """


class HeaderParser:

    def __init__(self) -> None:
        import re

        # [kernel_name, c signature]
        self.linker_directives = re.compile(r"//[\s]*tt-linker:[\s]*([\w]+):(.+):(.+)")
        # [name]
        self.kernel_name = re.compile("^([\\w]+)$")
        # [(type, name)]
        self.c_sig = re.compile(r"[\s]*([\w\*]+)\s([\w\*]+)[,]?")
        # [d|c]
        self.arg_suffix = re.compile("[c,d]")

        self.kernels = defaultdict(list)

    def extract_linker_meta(self, header: str):
        kernels_info = []
        for ln in header.splitlines():
            if ln.startswith("//"):
                m = self.linker_directives.match(ln)
                if _exists(m):
                    ker_name, c_sig, jit_ker_name = m.group(1), m.group(2), m.group(3)
                    name = self._match_name(ker_name)
                    c_types, arg_names = self._match_c_sig(c_sig)
                    kernels_info.append({
                        'name': name,
                        'arg_names': arg_names,
                        'c_types': c_types,
                        'jit_ker_name': jit_ker_name,
                    })
        return kernels_info

    def _match_name(self, ker_name: str):
        m = self.kernel_name.match(ker_name)
        if _exists(m):
            name = m.group(1)
            return name
        raise LinkerError(f"{ker_name} is not a valid kernel name")

    def _match_c_sig(self, c_sig: str):
        m = self.c_sig.findall(c_sig)
        if len(m):
            tys, args = [], []
            for ty, arg_name in m:
                tys.append(ty)
                args.append(arg_name)
            return tys, args

        raise LinkerError(f"{c_sig} is not a valid argument signature")

    def _add_kernel(self, name: str, ker: KernelLinkerMeta):
        if name in self.kernels:
            last: KernelLinkerMeta = self.kernels[name][-1]

            for cur, new_ in zip(last.arg_ctypes, ker.arg_ctypes):
                if cur != new_:
                    raise LinkerError(
                        f"Mismatched signature for kernel {name}: \n\texisting sig is: {','.join(last.arg_ctypes)}\n\tcurrent is: {','.join(ker.arg_ctypes)}"
                    )

        self.kernels[name].append(ker)


def gen_signature_with_full_args(m):
    return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])


def gen_signature(m):
    arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
    arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
    sig = ", ".join([f"{ty} {arg}" for ty, arg in zip(arg_types, arg_names)])
    return sig


def make_global_decl(meta: KernelLinkerMeta) -> str:
    """Generate declarations of kernels with meta-parameter and constant values."""
    return f"""
cnrtRet_t {meta.orig_kernel_name}_default(cnrtQueue_t queue, cnrtDim3_t* dim, {gen_signature_with_full_args(meta)});
cnrtRet_t {meta.orig_kernel_name}(cnrtQueue_t queue, cnrtDim3_t* dim, {gen_signature_with_full_args(meta)}, int algo_id);
"""


def make_default_algo_kernel(meta: KernelLinkerMeta) -> str:
    """Generate dispatcher function for kernels with different meta-parameter and constant values."""
    src = f"cnrtRet_t {meta.orig_kernel_name}_default(cnrtQueue_t queue, cnrtDim3_t* dim, {gen_signature_with_full_args(meta)}){{\n"
    src += (f"  return {meta.orig_kernel_name}(queue, dim, {', '.join(meta.arg_names)}, 0);\n")
    src += "}\n"
    return src


def make_kernel_meta_const_dispatcher(meta: KernelLinkerMeta) -> str:
    """Generate dispatcher function for kernels with different meta-parameter and constant values."""
    src = f"cnrtRet_t {meta.orig_kernel_name}(cnrtQueue_t queue, cnrtDim3_t* dim, {gen_signature_with_full_args(meta)}, int algo_id){{\n"
    src += f"  int algo_num = (int)(sizeof({meta.orig_kernel_name}_kernels) / sizeof({meta.orig_kernel_name}_kernels[0]));\n"
    src += f"  algo_id = (algo_id >=0 && algo_id < algo_num) ? algo_id : 0;\n"
    src += f"  return {meta.orig_kernel_name}_kernels[algo_id](queue, dim, {', '.join(meta.arg_names)});\n"
    src += "}\n"
    return src


def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
    """Generate definition of function pointers of kernel dispatchers based on meta-parameter and constant values."""
    # the table of hint dispatchers
    src = ""
    for name in names:
        src += f"cnrtRet_t {name}(cnrtQueue_t queue, cnrtDim3_t* dim, {gen_signature_with_full_args(meta)});\n"
    src += f"typedef cnrtRet_t (*kernel_func_t)(cnrtQueue_t queue, cnrtDim3_t* dim, {gen_signature_with_full_args(meta)});\n"
    src += f"kernel_func_t {meta.orig_kernel_name}_kernels[] = {{\n"
    for name in names:
        src += f"  {name},\n"
    src += "};\n"
    return src


def make_get_num_algos_decl(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
    return src


def make_get_num_algos_def(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
    src += f"  return (int)(sizeof({meta.orig_kernel_name}_kernels) / sizeof({meta.orig_kernel_name}_kernels[0]));\n"
    src += "}\n"
    return src


desc = """
Triton ahead-of-time linker:

This program takes in header files generated by mlu_compile.py, and generates a
single entry-point responsible for dispatching the user's input to the right
kernel given the specializations that were compiled.

Example usage:
python mlu_link.py mod_kernel_0_cc_version.h mod_kernel_1_cc_version.h \
    mod_kernel_2_cc_version.h -o all_in_one

The above command generates two files, `all_in_one.c` and `all_in_one.h`, which
include the following three functions:

// Get the number of algorithm implementations for mod_kernel
int mod_kernel_get_num_algos(void);

// Launch the default implementation (algo_id=0) of mod_kernel
cnrtRet_t mod_kernel_default(cnrtQueue_t queue, cnrtDim3_t* dim, void* Z, void* X, void* Y);

// Launch the mod_kernel implementation by algo_id
cnrtRet_t mod_kernel(cnrtQueue_t queue, cnrtDim3_t* dim, void* Z, void* X, void* Y, int algo_id);

Notes:
1. The header files passed to `mlu_link.py` must belong entirely to different
   implementations of the same kernel (Different versions of jit kernels from
   one Python file, e.g. with different `BLOCK_SIZE`, or jit kernels from a
   series of different Python files with the same jit kernel name). Do not
   pass a collection of headers that contain multiple unrelated kernels.

2. Different versions of the same kernel must have identical function parameter
   lists. That means both the number of parameters and the type of each parameter
   at every position must remain consistent.

3. When selecting a specific kernel by algo_id, if algo_id is less than 0 or
   greater than the maximum valid value, the kernel with algo_id equal to 0
   will be used by default.
"""

if __name__ == "__main__":
    parser = ArgumentParser(description=desc, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "headers",
        nargs="+",
        help="Paths to header files to link. Must include linker directive annotations (autogenerated by ttc)",
    )
    parser.add_argument("--out", "-o", type=Path, help="Out filename")
    args = parser.parse_args()

    # metadata
    parser = HeaderParser()
    includes = []
    kernels_info = []
    header_kernels_range = {}
    for header in args.headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        cur_header_kernels = parser.extract_linker_meta(h_str)
        header_kernels_range[header] = [len(kernels_info), len(cur_header_kernels)]
        kernels_info.extend(cur_header_kernels)
    jit_kernel_names = [kwarg['jit_ker_name'] for kwarg in kernels_info]
    kernel_names = [kwarg['name'] for kwarg in kernels_info]
    kernel_c_types = [kwarg['c_types'] for kwarg in kernels_info]
    kernel_num = len(jit_kernel_names)
    if kernel_num < 1:
        raise LinkerError(f"The linking process requires at least one kernels, but there are currently {kernel_num}.")
    for idx, name in enumerate(jit_kernel_names[1:], start=1):
        if name == jit_kernel_names[0]:
            continue
        for k, v in header_kernels_range.items():
            if idx >= v[0] and idx < v[0] + v[1]:
                invalid_header = k
        raise LinkerError(f"The kernel function symbol '{kernel_names[idx]}' in the file '{invalid_header}', " +
                          f"which corresponds to the JIT kernel '{jit_kernel_names[idx]}', is incompatible " +
                          f"with the first kernel function '{kernel_names[0]}' in the file '{args.headers[0]}', " +
                          f"which corresponds to the JIT kernel '{jit_kernel_names[0]}'. " +
                          f"All JIT kernels should have the same name.")
    for idx in range(kernel_num):
        signature_is_compatible = True
        if len(kernel_c_types[idx]) != len(kernels_info[0]['c_types']):
            signature_is_compatible = False
        else:
            for ctype_first_func, ctype_latter_func in zip(kernels_info[0]['c_types'], kernel_c_types[idx]):
                if ctype_first_func != ctype_latter_func:
                    signature_is_compatible = False
        if not signature_is_compatible:
            for k, v in header_kernels_range.items():
                if idx >= v[0] and idx < v[0] + v[1]:
                    invalid_header = k
            raise LinkerError(f"The kernel function symbol '{kernel_names[idx]}' in the file '{invalid_header}', " +
                              f"which has the sinature \"{kernel_c_types[idx]}\", is incompatible " +
                              f"with the first kernel function '{kernel_names[0]}' in the file '{args.headers[0]}', " +
                              f"which has the sinature \"{kernel_c_types[0]}\". " +
                              f"All kernel function should have the same sinature.")
            raise LinkerError(
                f"The function \"{kernel_names[idx]}\" 's signature  from file \"{invalid_header}\" has an incompatible name with the first function \"{kernels_info[0]['c_types']}\"."
            )
        parser._add_kernel(
            kernel_names[idx],
            KernelLinkerMeta(orig_kernel_name=jit_kernel_names[0], arg_names=kernels_info[idx]['arg_names'],
                             arg_ctypes=kernel_c_types[idx]),
        )

    # generate headers
    meta_lists = [meta for name, meta in parser.kernels.items()]
    meta = meta_lists[0][0]
    get_num_algos_decl = make_get_num_algos_decl(meta)
    global_decl = make_global_decl(meta)
    with args.out.with_suffix(".h").open("w") as fp:
        out = "#include <cnrt.h>\n"
        out += "\n"
        out += "#ifdef __cplusplus\n"
        out += "extern \"C\" {\n"
        out += "#endif\n"
        out += "\n"
        out += get_num_algos_decl
        out += "\n"
        out += global_decl
        out += "\n"
        out += "#ifdef __cplusplus\n"
        out += "}\n"
        out += "#endif\n"
        fp.write(out)

    # generate source
    names = [name for name in parser.kernels.keys()]
    func_pointers_def = make_func_pointers(names, meta)
    meta_const_def = make_kernel_meta_const_dispatcher(meta)
    get_num_algos_def = make_get_num_algos_def(meta)
    default_algo_kernel = make_default_algo_kernel(meta)
    with args.out.with_suffix(".c").open("w") as fp:
        out = ""
        out += "#include <cnrt.h>\n"
        out += "#include <stdint.h>\n"
        out += "#include <assert.h>\n"
        out += "\n"
        out += "#ifdef __cplusplus\n"
        out += "extern \"C\" {\n"
        out += "#endif\n"
        out += "\n"
        out += func_pointers_def
        out += "\n"
        out += get_num_algos_def
        out += "\n"
        out += meta_const_def
        out += "\n"
        out += default_algo_kernel
        out += "\n"
        out += "#ifdef __cplusplus\n"
        out += "}\n"
        out += "#endif\n"
        fp.write(out)
