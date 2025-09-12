import numpy as np


def generate_linear_decreasing_sparsity_nonzero(S, L, min_value=0.1):
    """
    生成线性递减的稀疏度列表，使第一个稀疏度为1，最后一个稀疏度不为0，平均稀疏度为S。

    参数:
    - S: float, 平均稀疏度 (0 < S <= 1)
    - L: int, 层数 (L > 1)
    - min_value: float, 最小稀疏度值 (0 <= min_value < 1)

    返回:
    - list: 稀疏度列表
    """
    if not (0 < S <= 1):
        raise ValueError("平均稀疏度 S 必须在 (0, 1] 之间")
    if L <= 1:
        raise ValueError("层数 L 必须大于 1")
    if not (0 <= min_value < 1):
        raise ValueError("最小稀疏度 min_value 必须在 [0, 1) 之间")

    if S == min_value:
        return [min_value] * L

    # # 生成线性递减的初始稀疏度序列，从 1 到 min_value
    # initial_sparsity = np.linspace(1, min_value, L)

    # # 计算初始平均值
    # initial_mean = np.mean(initial_sparsity)

    # # 调整比例使得平均稀疏度为 S
    # scaling_factor = S / initial_mean
    # adjusted_sparsity = initial_sparsity * scaling_factor

    # # 确保稀疏度在 [min_value, 1] 范围内
    # adjusted_sparsity = np.clip(adjusted_sparsity, min_value, 1)

    # return adjusted_sparsity.tolist()

    # 生成指数递减的稀疏度序列
    x = np.linspace(0, 1, L)
    sparsity_sequence = np.exp(-1 * x)  # 使用指数衰减函数

    # 将稀疏度缩放至 [min_value, 1] 区间
    sparsity_sequence = (sparsity_sequence - sparsity_sequence.min()) / (sparsity_sequence.max() -
                                                                         sparsity_sequence.min())
    sparsity_sequence = sparsity_sequence * (1 - min_value) + min_value

    # 计算初始平均值
    initial_mean = np.mean(sparsity_sequence)

    # 调整比例使得平均稀疏度为 S
    scaling_factor = S / initial_mean
    adjusted_sparsity = sparsity_sequence * scaling_factor

    # 确保稀疏度在 [min_value, 1] 范围内
    adjusted_sparsity = np.clip(adjusted_sparsity, min_value, 1)

    return adjusted_sparsity.tolist()


# 测试
S = 0.125  # 目标平均稀疏度
L = 32  # 层数
min_value = 0.0625  # 最小稀疏度值
sparsity_list = generate_linear_decreasing_sparsity_nonzero(S, L, min_value)
print("生成的稀疏度列表:", sparsity_list)
print("平均稀疏度:", np.mean(sparsity_list))
