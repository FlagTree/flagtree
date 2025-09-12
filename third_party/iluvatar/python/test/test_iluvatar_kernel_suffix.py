import os
from triton.compiler.code_generator import kernel_suffix


def test_kernel_suffix():
    # Construct signature and Specialization
    signature = ['a', 'b', 'c', 'd']

    class Specialization:
        equal_to_1 = {1}
        divisible_by_16 = {2}
        divisible_by_8 = {3}

    # Get result and check
    result = kernel_suffix(signature, Specialization())
    print(f"[kernel_suffix] Result: {result}")
    assert result == "01c2d3e", f"Unexpected result: {result}"


if __name__ == "__main__":
    print("== Running test_kernel_suffix ==")
    test_kernel_suffix()
