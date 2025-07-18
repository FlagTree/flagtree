# TODO: 0624
import os
from triton.backend_loader import get_backend


def test_kernel_suffix(platform: str):
    backend = get_backend(platform)
    
    # 构造 signature 和 specialization
    signature = ['a', 'b', 'c', 'd']

    class Specialization:
        equal_to_1 = {1}
        divisible_by_16 = {2}
        divisible_by_8 = {3}

    result = backend.kernel_suffix(signature, Specialization())
    print(f"[kernel_suffix] Result: {result}")
    assert result == "01c2d3e", f"Unexpected result: {result}"


if __name__ == "__main__":
    # 设置平台（你可以根据实际需要改为 "platformA"、"iluvatar" 等）
    PLATFORM = os.getenv("FLAGTREE_PLATFORM", "iluvatar")

    print("== Running test_kernel_suffix ==")
    test_kernel_suffix(PLATFORM)

