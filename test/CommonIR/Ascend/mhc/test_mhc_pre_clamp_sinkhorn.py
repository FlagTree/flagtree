"""Minimal runner for mhc_pre_clamp_sinkhorn.

Usage:
    python test_mhc_pre_clamp_sinkhorn.py
    python test_mhc_pre_clamp_sinkhorn.py --device npu --B 1 --S 16 --D 128
    python test_mhc_pre_clamp_sinkhorn.py --check
    MLIR_ENABLE_DUMP=1 python test_mhc_pre_clamp_sinkhorn.py --dump

This script constructs one valid input set and runs mhc_pre_clamp_sinkhorn,
optionally comparing the result against the matching PyTorch reference.
"""

from __future__ import annotations

import argparse
import os

import torch

from mhc_pre_clamp_sinkhorn import mhc_pre_clamp_sinkhorn, mhc_pre_clamp_sinkhorn_ref


def _resolve_device(device: str) -> str:
    if device == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _sync(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _make_inputs(B: int, S: int, N: int, D: int, dtype: torch.dtype, device: str):
    assert N == 4, "mhc_pre_clamp_sinkhorn currently requires N=4"
    hc_mix = N * (N + 2)
    hc_d = N * D

    x = torch.randn((B, S, N, D), dtype=dtype, device=device)
    phi = torch.randn((hc_mix, hc_d), dtype=torch.float32, device=device)
    alpha = torch.randn((3, ), dtype=torch.float32, device=device)
    base = torch.randn((hc_mix, ), dtype=torch.float32, device=device)
    return x, phi, alpha, base


def _check(out, ref):
    y_ok = torch.allclose(out["y"].float(), ref["y"].float(), atol=1e-2, rtol=1e-2)
    post_ok = torch.allclose(out["post_out"].float(), ref["post_out"].float(), atol=1e-4, rtol=1e-4)
    comb_ok = torch.allclose(out["comb_frag"].float(), ref["comb_frag"].float(), atol=1e-4, rtol=1e-4)
    return bool(y_ok and post_ok and comb_ok), (bool(y_ok), bool(post_ok), bool(comb_ok))


def main() -> None:
    parser = argparse.ArgumentParser(description="mhc_pre_clamp_sinkhorn runner")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto/npu/cuda/cpu")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--S", type=int, default=8, help="Sequence length")
    parser.add_argument("--N", type=int, default=4, help="Head multiplier; must be 4")
    parser.add_argument("--D", type=int, default=128, help="Hidden dimension per head")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Input dtype for x")
    parser.add_argument("--iter-times", type=int, default=20, help="Sinkhorn iteration count")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=0.0)
    parser.add_argument("--need-backward", action="store_true", help="Request extra saved tensors")
    parser.add_argument("--check", action="store_true", help="Compare the result to the reference")
    parser.add_argument("--dump", action="store_true", help="Print current MLIR_ENABLE_DUMP setting as a reminder")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.dump:
        print(f"MLIR_ENABLE_DUMP={os.environ.get('MLIR_ENABLE_DUMP', '<unset>')}")

    print(f"device={device} shape=({args.B}, {args.S}, {args.N}, {args.D}) dtype={args.dtype} check={args.check}")

    x, phi, alpha, base = _make_inputs(args.B, args.S, args.N, args.D, dtype, device)
    kwargs = dict(
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
    )

    try:
        out = mhc_pre_clamp_sinkhorn(x, phi, alpha, base, need_backward=args.need_backward, **kwargs)
        _sync(device)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    if args.check:
        ref = mhc_pre_clamp_sinkhorn_ref(x, phi, alpha, base, **kwargs)
        ok, detail = _check(out, ref)
        print(f"[{'PASS' if ok else 'FAIL'}] y={detail[0]} post={detail[1]} comb={detail[2]}")
        if not ok:
            raise SystemExit(1)
    else:
        print(f"[ok] y{tuple(out['y'].shape)} "
              f"post{tuple(out['post_out'].shape)} comb{tuple(out['comb_frag'].shape)}")
        if args.need_backward:
            for key in ["inv_rms", "x_scaled", "mixes", "h_res_logits", "pre"]:
                print(f"       {key}{tuple(out[key].shape)} dtype={out[key].dtype}")


if __name__ == "__main__":
    main()
