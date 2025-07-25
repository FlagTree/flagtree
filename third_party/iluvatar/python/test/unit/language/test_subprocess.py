import itertools
import os
import subprocess
import sys
from collections import Counter
from triton.runtime.build import is_corex

import pytest
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
print_path = os.path.join(dir_path, "print_helper.py")
assert_path = os.path.join(dir_path, "assert_helper.py")

# TODO: bfloat16 after LLVM-15
assert_types = ["device_assert", "device_assert_passes", "assert", "static_assert", "no_debug", "double_assert"]
nested_types = [(caller, callee) for caller in ["true", "false", "none"] for callee in ["true", "false", "none"]]
if is_corex():
    torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32"]
else:
    torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


# TODO: Print with multiple operands


@pytest.mark.interpreter
@pytest.mark.parametrize("func_type, data_type", [("device_print", data_type) for data_type in torch_types] + [
    ("print", "int32"),
    ("static_print", "int32"),
    ("no_arg_print", "int32"),
    ("print_no_arg", "int32"),
    ("device_print_large", "int32"),
    ("print_multiple_args", "int32"),
    ("device_print_multiple_args", "int32"),
    ("device_print_hex", "int16"),
    ("device_print_hex", "int32"),
    ("device_print_hex", "int64"),
    ("device_print_pointer", "int32"),
])
def test_print(func_type: str, data_type: str):
    capability = torch.cuda.get_device_capability()
    if capability[0] == 8 and data_type in ["float16", "float32"]:
        pytest.skip("QS need vprint3.")
    proc = subprocess.Popen([sys.executable, print_path, func_type, data_type], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=False)
    outs, err = proc.communicate()
    assert proc.returncode == 0

    if is_interpreter() and func_type != "static_assert":
        # Interpreter uses a different format for device_print
        # Only check if there's no error
        assert err == b''
        return

    outs = [line for line in outs.decode("UTF-8").split("\n") if line]
    # The total number of elements in the 1-D tensor to print.
    N = 128

    # Format is
    #   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...) <prefix> (operand <n>) <elem>
    expected_lines = Counter()
    if func_type == "print" or func_type == "device_print":
        for i in range(N):
            line = f"pid (0, 0, 0) idx ({i:3}) x: {i}"
            if data_type.startswith("float"):
                line += ".000000"
            expected_lines[line] = 1
    elif func_type == "device_print_hex":
        for i in range(N):
            line = f"pid (0, 0, 0) idx ({i:3}) x: 0x"
            if data_type == "int16":
                line += f"{i:04x}"
            if data_type == "int32":
                line += f"{i:08x}"
            if data_type == "int64":
                line += f"{i:016x}"
            expected_lines[line] = 1
    elif func_type == "static_print":
        expected_lines[f" int32[constexpr[{N}]]"] = 1
    elif func_type == "no_arg_print":
        expected_lines["pid (0, 0, 0) idx (): 0"] = N
    elif func_type == "print_no_arg":
        expected_lines["pid (0, 0, 0) no arg"] = N
    elif func_type == "device_print_large":
        for i, j, k in itertools.product(range(2), range(64), range(N)):
            expected_lines[f"pid (0, {i}, 0) idx ({j:2}, {k:3}) x: 1"] = 1
    elif func_type == "print_multiple_args" or func_type == "device_print_multiple_args":
        for i in range(N):
            expected_lines[f"pid (0, 0, 0) idx ({i:3}): (operand 0) {i}"] = 1
            expected_lines[f"pid (0, 0, 0) idx ({i:3}): (operand 1) 1"] = 1
    elif func_type == "device_print_pointer":
        for i in range(N):
            expected_lines[f"pid (0, 0, 0) idx ({i:3}) ptr: 0x"] = 1

    actual_lines = Counter()
    for line in outs:
        # Trim the exact pointer address in the output--they can change per run.
        line = (line.split(':')[0] + ": 0x") if func_type == "device_print_pointer" else line
        actual_lines[line] += 1

    diff = Counter(actual_lines)
    diff.subtract(expected_lines)
    for line, delta in diff.items():
        if delta == 0:
            continue
        print(f'Expected line "{line}" {expected_lines[line]} time(s), but saw {actual_lines[line]} time(s)')
    assert all(delta == 0 for delta in diff.values())


@pytest.mark.parametrize("func_type", assert_types)
def test_assert(func_type: str):
    # The total number of elements in the 1-D tensor to assert on.
    N = 128

    os.environ["TRITON_DEBUG"] = "1"
    proc = subprocess.Popen([sys.executable, assert_path, func_type], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=False)
    # _, errs = proc.communicate()
    outs, errs = proc.communicate()
    errs = outs if is_corex() else errs
    errs = errs.splitlines()
    num_errs = 0
    for err in errs:
        if "x != 0" in err.decode("utf-8", errors="ignore"):
            num_errs += 1

    # Check for segfaults.
    assert all("segmentation fault" not in line.decode("utf-8", errors="ignore").lower() for line in errs)

    os.environ["TRITON_DEBUG"] = "0"
    if func_type == "static_assert" or func_type == "device_assert_passes":
        assert num_errs == 0
    else:
        assert num_errs == N - 1


@pytest.mark.parametrize("caller_type, callee_type", nested_types)
def test_assert_nested(caller_type, callee_type):
    # The total number of elements in the 1-D tensor to assert on.
    N = 128

    proc = subprocess.Popen([sys.executable, assert_path, caller_type, callee_type], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=False)
    # _, errs = proc.communicate()
    outs, errs = proc.communicate()
    errs = outs if is_corex() else errs
    errs = errs.splitlines()
    num_errs = 0
    for err in errs:
        if "x != 0" in err.decode("utf-8", errors="ignore"):
            num_errs += 1
    if caller_type == "none":
        if callee_type == "true":
            assert num_errs == N - 1
        else:
            assert num_errs == 0
    elif caller_type == "true":
        if callee_type == "false":
            assert num_errs == 0
        else:
            assert num_errs == N - 1
    elif caller_type == "false":
        if callee_type == "true":
            assert num_errs == N - 1
        else:
            assert num_errs == 0
