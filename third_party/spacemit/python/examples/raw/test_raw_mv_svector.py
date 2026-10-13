"""smt_rvisa svector-level mv (feishu 3.3 示例).

C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

style2 (纯 svector): vconfig/vzero/vload/vmacc/vreduce_sum/vstore, no packing.
style3 (svector + pack): the same, but B's 4-row block is pre-packed into a
        contiguous scratch buffer via sr.alloc + sr.pack before the K loop.

(矩阵单元 mv/mm 走 sr.vmadot → vector_ext.cross_batch_matmul,见
 test_raw_mm_cbm.py / test_raw_mv_cbm.py。)

Fixed VL (f16 -> 64) this round: no dynamic vsetvl tail handling, so the tests
constrain K % 64 == 0 and N % 4 == 0 (full tiles only).

访存按 SPEC §6.2 规格:vload(ptr, index)/vstore(ptr, index, value),index 为扁平
标量元素偏移,二维坐标由用户自行压平(如 B 的行 ni 列 ki 写作 ni*K + ki)。对
alloc 出的 ranked scratch(packed_B),index 仍是逐维下标元组(ranked 自然寻址)。
"""

import torch
import triton
import triton.language as tl
from triton.backends.spacemit.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f16 = sr.f16
f32 = sr.f32


# ---------------------------------------------------------------------------
# 写法2 — 纯 svector
# ---------------------------------------------------------------------------
@sr.raw_kernel
def mv_block_style2(B: sr.mem(f16), A: sr.mem(f16), C: sr.mem(f32, out=True), K: sr.index, row_base: sr.index,
                    row_end: sr.index):
    # grid 并发:host 按 program_id 把 N 行切块,本 program 只算 [row_base, row_end) 行。
    nvl = sr.vconfig(-1, 1)  # lmul=1 → VLMAX=64 (f16, SPEC §6.1)
    # strip-mine:K 循环分裂成主循环(满 tile)+尾循环(K%VL 那块)。要不要 pad 由 codegen 编译期
    # 按「是否设了 vconfig 收窄」决定(纯 Python if, 不往 IR 塞运行期 scf.if):主循环不收窄 →
    # vload 走快路直读, 零 pad 零分支;尾循环设 vconfig(K-ki) → vload fill-0 pad。尾循环
    # scf.for 天然跑 0 次(K%VL==0, 整除 shape)或 1 次, 用迭代次数代替分支。
    Kfloor = (K // nvl) * nvl  # 满 tile 覆盖的 K 区间(VL 整数倍)
    for ni in sr.range(row_base, row_end, 4):
        acc0 = sr.vzero(f32)
        acc1 = sr.vzero(f32)
        acc2 = sr.vzero(f32)
        acc3 = sr.vzero(f32)
        for ki in sr.range(0, Kfloor, nvl):  # 主循环:满 tile, 快路直读(不设收窄)
            va = sr.vload(A, ki)
            vb0 = sr.vload(B, ni * K + ki)
            vb1 = sr.vload(B, (ni + 1) * K + ki)
            vb2 = sr.vload(B, (ni + 2) * K + ki)
            vb3 = sr.vload(B, (ni + 3) * K + ki)
            acc0 = sr.vmacc(acc0, vb0, va)
            acc1 = sr.vmacc(acc1, vb1, va)
            acc2 = sr.vmacc(acc2, vb2, va)
            acc3 = sr.vmacc(acc3, vb3, va)
        for ki in sr.range(Kfloor, K, nvl):  # 尾循环:跑 0/1 次, 收窄 → codegen 走 fill-0 pad
            nvl = sr.vconfig(K - ki, 1)
            # 用独立临时名(ta/tb*):与主循环的 va/vb* 不同名, 否则它们泄漏到外层作用域,
            # 尾循环的 iter_arg 检测(_find_reassigned)会误把这些纯临时当成循环携带值 →
            # 生成引用主循环已出作用域 SSA 的坏 iter_args。
            ta = sr.vload(A, ki)
            tb0 = sr.vload(B, ni * K + ki)
            tb1 = sr.vload(B, (ni + 1) * K + ki)
            tb2 = sr.vload(B, (ni + 2) * K + ki)
            tb3 = sr.vload(B, (ni + 3) * K + ki)
            acc0 = sr.vmacc(acc0, tb0, ta)
            acc1 = sr.vmacc(acc1, tb1, ta)
            acc2 = sr.vmacc(acc2, tb2, ta)
            acc3 = sr.vmacc(acc3, tb3, ta)
        sr.sstore(C, ni, sr.vreduce_sum(acc0))
        sr.sstore(C, ni + 1, sr.vreduce_sum(acc1))
        sr.sstore(C, ni + 2, sr.vreduce_sum(acc2))
        sr.sstore(C, ni + 3, sr.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style2(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)  # 限行:末 program 不越 N, 避免读 phantom 行 B
    _sr_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, row_base, row_end])


# ---------------------------------------------------------------------------
# 写法3 — svector + sr.alloc/sr.pack 预打包 B
# ---------------------------------------------------------------------------
@sr.raw_kernel
def mv_block_style3(B: sr.mem(f16), A: sr.mem(f16), C: sr.mem(f32, out=True), K: sr.index, row_base: sr.index,
                    row_end: sr.index):
    # grid 并发:本 program 只算 [row_base, row_end) 行。
    nvl = sr.vconfig(-1, 1)  # lmul=1 → VLMAX=64 (f16, SPEC §6.1)
    # tile 数须 ceil(K/nvl):pack 的 kb 循环跑 0..K step nvl = ceil(K/nvl) 个 tile,
    # alloc 用 floor(K//nvl)在 K%64≠0 时欠分配 → pack 写 dst[0,ceil-1,..] 越界 dim-1
    # → 堆缓冲区溢出(跨-kernel 污染, NaN)。用 ceil 匹配 pack 实际写的 tile 数。
    kct = (K + nvl - 1) // nvl
    packed_B = sr.alloc((1, kct, 4, nvl), f16)
    # strip-mine(同 style2):主循环满 tile 走 vload 快路(不设收窄)、尾循环 K%VL 那块设
    # vconfig(K-ki)→codegen 编译期走 fill-0 pad。要不要 pad 由 codegen 按是否收窄编译期定,
    # 不塞运行期 scf.if;尾循环 scf.for 跑 0/1 次代替分支。packed_B 尾 tile 由 pack 已补 0。
    Kfloor = (K // nvl) * nvl
    for ni in sr.range(row_base, row_end, 4):
        acc0 = sr.vzero(f32)
        acc1 = sr.vzero(f32)
        acc2 = sr.vzero(f32)
        acc3 = sr.vzero(f32)
        sr.pack(B, (ni, 0), packed_B, (1, kct, 4, nvl), K)
        for ki in sr.range(0, Kfloor, nvl):  # 主循环:满 tile 快路
            vb0 = sr.vload(packed_B, (0, ki // nvl, 0, 0))
            vb1 = sr.vload(packed_B, (0, ki // nvl, 1, 0))
            vb2 = sr.vload(packed_B, (0, ki // nvl, 2, 0))
            vb3 = sr.vload(packed_B, (0, ki // nvl, 3, 0))
            va = sr.vload(A, ki)
            acc0 = sr.vmacc(acc0, vb0, va)
            acc1 = sr.vmacc(acc1, vb1, va)
            acc2 = sr.vmacc(acc2, vb2, va)
            acc3 = sr.vmacc(acc3, vb3, va)
        for ki in sr.range(Kfloor, K, nvl):  # 尾循环:跑 0/1 次, 收窄→fill-0(独立临时名 tb*/ta)
            nvl = sr.vconfig(K - ki, 1)
            tb0 = sr.vload(packed_B, (0, ki // nvl, 0, 0))
            tb1 = sr.vload(packed_B, (0, ki // nvl, 1, 0))
            tb2 = sr.vload(packed_B, (0, ki // nvl, 2, 0))
            tb3 = sr.vload(packed_B, (0, ki // nvl, 3, 0))
            ta = sr.vload(A, ki)
            acc0 = sr.vmacc(acc0, tb0, ta)
            acc1 = sr.vmacc(acc1, tb1, ta)
            acc2 = sr.vmacc(acc2, tb2, ta)
            acc3 = sr.vmacc(acc3, tb3, ta)
        sr.sstore(C, ni, sr.vreduce_sum(acc0))
        sr.sstore(C, ni + 1, sr.vreduce_sum(acc1))
        sr.sstore(C, ni + 2, sr.vreduce_sum(acc2))
        sr.sstore(C, ni + 3, sr.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style3(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)  # 限行:末 program 不越 N, pack 不读 phantom 行 B
    _sr_call(mv_block_style3, outputs=[], inputs=[B, A, C, K, row_base, row_end])


def _run(host, N, K, BLOCK=4):
    # grid 并发:N 行按 BLOCK 切成 ceil(N/BLOCK) 个 program(每个算 BLOCK 行,内层 4 行一组)。
    # N%BLOCK≠0 时末 program 的 row_end 会超过 N,内层无条件 vstore(C, ni..ni+3) 会写
    # phantom 行 C[N..Np-1]。C 须分配到 Np=ceil(N/BLOCK)*BLOCK 行吸收这些写(仅多分配,
    # 无 data copy),否则写越界 → 堆损坏(fix:N 尾 phantom 行越界)。取结果切 C[:N]。
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    grid = (Np // BLOCK, )
    host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    got = C[:N]
    ref = torch.mv(B.float(), A.float())
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"


_SHAPES = [(4, 64), (8, 128), (16, 256), (32, 512), (64, 64), (128, 256)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style2(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style3(N, K):
    _run(_mv_sv_host_style3, N, K)


# ---------------------------------------------------------------------------
# 任意 shape — 全 kernel 内 padding(零 host copy)
# ---------------------------------------------------------------------------
# 两个约束都在 kernel/host 封装内解决,B/A 不做任何 host copy:
#   ① K%64:K 尾块 vload 由 vconfig(K-ki,1) 降级为 fill-0 scratch,尾 lane 补 0
#      (probe_svpad_fill 坐实),vmacc/vreduce_sum 补 0 无害。
#   ② N%BLOCK:grid=ceil(N/BLOCK),末 program row_end>N,内层无条件写 phantom 行
#      C[N..Np-1];C 分配到 Np=ceil(N/BLOCK)*BLOCK 吸收(仅分配无 copy),切 C[:N]。
# _run 已同时处理 ①②,故任意 shape 直接复用 _run。
# 任意 shape:K 非 64 倍数 / N 非 BLOCK 倍数 / 二者都非整除
_SHAPES_ARB = [(4, 60), (4, 100), (8, 65), (4, 63), (8, 127), (4, 200), (16, 130), (12, 50), (7, 64), (33, 65),
               (50, 130), (100, 100), (6, 60), (13, 200), (37, 130)]


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style2_arb(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style3_arb(N, K):
    _run(_mv_sv_host_style3, N, K)
