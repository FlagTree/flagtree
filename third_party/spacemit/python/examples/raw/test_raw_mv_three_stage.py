"""Parallel (multi-core) 3-stage mv: svector pre-scale -> call_intrinsic vfwmacc
-> svector post-scale, fused into ONE launch, partitioned across programs.

  stage 1  pre_scale_svector     : vec_s[k] = vec[k] * alpha   —— svector
  stage 2  mv_vfwmacc_ci_parallel: scores = Mat @ vec_s        —— call_intrinsic
                                    (only llvm.riscv.vfwmacc stays a real
                                    call_intrinsic; load/reduce/store are native)
  stage 3  post_scale_svector_par: out[base:bend] = scores * beta —— svector

Multi-core: host `_mv_fused_host_par_sv3` launches grid=(N//BLK,). Each program
computes base/bend from tl.program_id(0) and handles ONE BLK-row tile. stage 2's
sibling llvm.func resolves program_id via the runtime spine_grid(ctx, axis);
stage 3 needs no program_id in its body — the host passes base/bend as index
params (normal _sr_call allows computed values).

Constraints: K % 64 == 0, N % 8 == 0, BLK % 8 == 0, N % BLK == 0.
Run under pytest — `python file.py` re-triggers the do_not_specialize host
recompile quirk.
"""
import time
import torch
import triton
import triton.language as tl
from triton.backends.spacemit.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

_ALPHA = 1.5  # pre-scale  factor (module-level → svector constexpr_float)
_BETA = 0.5  # post-scale factor


# ── stage 1: svector pre-scale  vec_s = vec * alpha ─────────────────────────
# f16 vec → cast f32 → mul alpha (broadcast) → cast f16 → store. K % 64 == 0.
@tle.raw_kernel
def pre_scale_svector(vec: tle.mem(f16), vec_s: tle.mem(f16, out=True), K: tle.index):
    nvl = tle.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for ki in tle.range(0, Kfloor, nvl):
        v = tle.vload(vec, ki)  # vector<64xf16>
        v_f = tle.cast(v, f32)  # vector<64xf32>  (arith.extf)
        v_s = v_f * _ALPHA  # vector<64xf32>  (arith.mulf, scalar bcast)
        v_s_h = tle.cast(v_s, f16)  # vector<64xf16>  (arith.truncf)
        tle.vstore(vec_s, ki, v_s_h)
    # tail loop: distinct names — reusing v/v_f/v_s/v_s_h would make the codegen
    # treat them as cross-loop iter_args referencing SSA from the closed main loop.
    for ki in tle.range(Kfloor, K, nvl):
        nvl = tle.vconfig(K - ki, 1)
        tv = tle.vload(vec, ki)
        tv_f = tle.cast(tv, f32)
        tv_s = tv_f * _ALPHA
        tv_s_h = tle.cast(tv_s, f16)
        tle.vstore(vec_s, ki, tv_s_h)


# ── stage 2: 8-row vfwmacc mv, program_id-partitioned (grid=(N//BLK,)) ───────
# Pure llvm-direct (sibling llvm.func). Every op uses the SAME tle.call_intrinsic;
# the emitter renders llvm.load / llvm.vector.reduce.fadd / llvm.store as NATIVE
# llvm ops, so ONLY llvm.riscv.vfwmacc stays a real call_intrinsic. Each program
# handles its own BLK-row tile at row_base = program_id(0)*BLK — resolved inside
# the sibling via the runtime spine_grid(ctx, axis). BLK % 8 == 0, N % BLK == 0.
@tle.raw_kernel
def mv_vfwmacc_ci_parallel(
        B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, N: tle.index, BLK: tle.index):
    vl = tle.llvm_const(64, "i64")
    zero = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    cbase = tle.llvm_base_ptr(C)
    abase = tle.llvm_base_ptr(A)
    bbase = tle.llvm_base_ptr(B)

    row_base = tle.program_id(0) * BLK  # this program's BLK-row tile base
    row_end = row_base + BLK
    for ni in tle.range(row_base, row_end, 8):
        acc0 = zero_acc
        acc1 = zero_acc
        acc2 = zero_acc
        acc3 = zero_acc
        acc4 = zero_acc
        acc5 = zero_acc
        acc6 = zero_acc
        acc7 = zero_acc
        for ki in tle.range(zero, K, vl):
            ga = tle.llvm_gep(abase, ki, "f16")
            va = tle.call_intrinsic("llvm.load", [ga], result_type="vector<[4]xf16>")
            gb0 = tle.llvm_gep(bbase, ni * K + ki, "f16")
            gb1 = tle.llvm_gep(bbase, (ni + 1) * K + ki, "f16")
            gb2 = tle.llvm_gep(bbase, (ni + 2) * K + ki, "f16")
            gb3 = tle.llvm_gep(bbase, (ni + 3) * K + ki, "f16")
            gb4 = tle.llvm_gep(bbase, (ni + 4) * K + ki, "f16")
            gb5 = tle.llvm_gep(bbase, (ni + 5) * K + ki, "f16")
            gb6 = tle.llvm_gep(bbase, (ni + 6) * K + ki, "f16")
            gb7 = tle.llvm_gep(bbase, (ni + 7) * K + ki, "f16")
            vb0 = tle.call_intrinsic("llvm.load", [gb0], result_type="vector<[4]xf16>")
            vb1 = tle.call_intrinsic("llvm.load", [gb1], result_type="vector<[4]xf16>")
            vb2 = tle.call_intrinsic("llvm.load", [gb2], result_type="vector<[4]xf16>")
            vb3 = tle.call_intrinsic("llvm.load", [gb3], result_type="vector<[4]xf16>")
            vb4 = tle.call_intrinsic("llvm.load", [gb4], result_type="vector<[4]xf16>")
            vb5 = tle.call_intrinsic("llvm.load", [gb5], result_type="vector<[4]xf16>")
            vb6 = tle.call_intrinsic("llvm.load", [gb6], result_type="vector<[4]xf16>")
            vb7 = tle.call_intrinsic("llvm.load", [gb7], result_type="vector<[4]xf16>")
            acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc0, va, vb0, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc1, va, vb1, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc2 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc2, va, vb2, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc3 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc3, va, vb3, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc4 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc4, va, vb4, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc5 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc5, va, vb5, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc6 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc6, va, vb6, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
            acc7 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc7, va, vb7, zero, vl, zero],
                                      result_type="vector<[4]xf32>")
        s0 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32")
        s1 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32")
        s2 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32")
        s3 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32")
        s4 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32")
        s5 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32")
        s6 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32")
        s7 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32")
        tle.call_intrinsic("llvm.store", [s0, tle.llvm_gep(cbase, ni, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s1, tle.llvm_gep(cbase, ni + 1, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s2, tle.llvm_gep(cbase, ni + 2, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s3, tle.llvm_gep(cbase, ni + 3, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s4, tle.llvm_gep(cbase, ni + 4, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s5, tle.llvm_gep(cbase, ni + 5, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s6, tle.llvm_gep(cbase, ni + 6, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s7, tle.llvm_gep(cbase, ni + 7, "f32")], result_type="()")


# ── stage 3 (svector, parallel): out[base:bend] = scores[base:bend] * beta ───
# Pure svector (vload/mul/vstore) tiled variant. No program_id inside the body:
# the HOST computes base/bend from tl.program_id and passes them as plain index
# params (normal _sr_call allows computed values — only the mixed-mode llvm-direct
# _sr_call requires host entry-block args). dtype=f32 required on vload.
@tle.raw_kernel
def post_scale_svector_par(scores: tle.mem(f32), out: tle.mem(f32, out=True), base: tle.index, bend: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = ((bend - base) // nvl) * nvl + base
    for ni in tle.range(base, Nfloor, nvl):
        v = tle.vload(scores, ni, dtype=f32)
        v_s = v * _BETA
        tle.vstore(out, ni, v_s)
    for ni in tle.range(Nfloor, bend, nvl):  # tail — distinct SSA names
        nvl = tle.vconfig(bend - ni, 1)
        tv = tle.vload(scores, ni, dtype=f32)
        tv_s = tv * _BETA
        tle.vstore(out, ni, tv_s)


# ── parallel fused host, stage 3 via SVECTOR (grid=(N//BLK,)) ────────────────
# host computes base/bend from program_id and passes them as index params to the
# svector stage-3 kernel. stage 2 resolves its own program_id in the sibling.
@triton.jit(do_not_specialize=["K", "N", "BLK"])
def _mv_fused_host_par_sv3(Mat, vec, vec_s, scores, out, K, N, BLK):
    pid = tl.program_id(0)
    base = pid * BLK
    bend = base + BLK
    _sr_call(pre_scale_svector, outputs=[], inputs=[vec, vec_s, K])
    _sr_call(mv_vfwmacc_ci_parallel, outputs=[], inputs=[Mat, vec_s, scores, K, N, BLK])
    _sr_call(post_scale_svector_par, outputs=[], inputs=[scores, out, base, bend])


# ---------------------------------------------------------------------------
# Correctness + perf
# ---------------------------------------------------------------------------
_SHAPES = [(8, 64), (16, 128), (32, 256), (64, 512), (128, 256), (64, 64), (32, 128), (16, 64)]


def _run_fused_par_sv3_correctness(N, K, BLK=8):
    assert K % 64 == 0 and N % 8 == 0, "fused_par_sv3: K%64==0, N%8==0"
    assert BLK % 8 == 0 and N % BLK == 0, "fused_par_sv3: BLK%8==0, N%BLK==0"
    Np = ((N + 7) // 8) * 8
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)

    _mv_fused_host_par_sv3[(N // BLK, )](Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)

    got = out[:N]
    ref = torch.mv(Mat.float(), (vec.float() * _ALPHA).half().float()) * _BETA
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


def _measure_fused_par_sv3(N, K, BLK=8, iters=50, warmup=5):
    assert BLK % 8 == 0 and N % BLK == 0, "fused_par_sv3: BLK%8==0, N%BLK==0"
    Np = ((N + 7) // 8) * 8
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    grid = (N // BLK, )
    for _ in range(warmup):
        _mv_fused_host_par_sv3[grid](Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_fused_host_par_sv3[grid](Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_parallel_sv3_correctness(N, K):
    """Parallel fused, stage 3 via svector (host-computed base/bend), grid>1."""
    _run_fused_par_sv3_correctness(N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_parallel_sv3_perf(N, K):
    """Multi-core fused mv throughput (grid=(N//BLK,))."""
    BLK = 8
    t = _measure_fused_par_sv3(N, K, BLK=BLK)
    gf = 2.0 * N * K / t / 1e9
    print(f"N={N:4d} K={K:4d}  fused_par_sv3[BLK={BLK}]={t*1e6:8.1f}us ({gf:.2f}GF)")


if __name__ == "__main__":
    import sys

    failures = 0
    print("=== correctness: parallel sv3 (grid=(N//BLK,)) ===")
    for N, K in _SHAPES:
        try:
            md = _run_fused_par_sv3_correctness(N, K)
            print(f"  N={N:4d} K={K:4d}  max_diff={md:.4e}  PASS")
        except Exception as e:
            failures += 1
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== perf: parallel sv3 ===")
    for N, K in _SHAPES:
        try:
            t = _measure_fused_par_sv3(N, K, BLK=8)
            gf = 2.0 * N * K / t / 1e9
            print(f"  N={N:4d} K={K:4d}  {t*1e6:8.1f}us ({gf:.2f}GF)")
        except Exception as e:
            failures += 1
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    # 失败必须显式暴露: 非零退出, 否则 CI/脚本调用方会误判为通过
    sys.exit(1 if failures else 0)
