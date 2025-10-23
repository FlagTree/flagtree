import torch
import torch_npu
import triton
import triton.language as tl
import time

@triton.jit
def triton_foo(in_ptr0, in_ptr1, in_ptr2, out_ptr0, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr,
                BLOCK2: tl.constexpr,
                X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, R: tl.constexpr,
                Z_STRIDE: tl.constexpr, Y_STRIDE: tl.constexpr, X_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
                X_STRIDE1: tl.constexpr, Y_STRIDE1: tl.constexpr, Z_STRIDE1: tl.constexpr, R_STRIDE1: tl.constexpr,
                ):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    base1 = tl.arange(0, BLOCK1_SUB)
    base2 = tl.arange(0, BLOCK2)
    nsub: tl.constexpr = BLOCK1 // BLOCK1_SUB
    # loops1 : tl.constexpr =  nsub * Y * Z
    loops1: tl.constexpr = nsub
    loops2: tl.constexpr = R // BLOCK2

    for z in range(Z):
        for y in range(Y):
            for loop1 in range(loops1):
                off1 = loop1
                x = offset + (off1 * BLOCK1_SUB) + base1[:, None]
                x1 = offset + (off1 * BLOCK1_SUB) + base1[None, :]

                for loop2 in range(loops2):
                    r = loop2 * BLOCK2 + base2[None, :]
                    r1 = loop2 * BLOCK2 + base2[:, None]
                    tmp0 = tl.load(in_ptr0 + ( (R_STRIDE * r) +  (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp1 = tl.load(in_ptr1 + ( (R_STRIDE * r) +  (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp2 = tmp0 + tmp1

                    tmp8 = tl.load(in_ptr2 + (R_STRIDE1 * r + X_STRIDE1 * x  + (Y_STRIDE1 * y) + (Z_STRIDE1 * z)), None)
                    tmp9 =  tmp8 + tmp2
                    tl.store(out_ptr0 +  (R_STRIDE1 * r + X_STRIDE1 * x  + (Y_STRIDE1 * y) + (Z_STRIDE1 * z)), tmp9,
                             None)

def foo_triton_wrapper(a, b, c):
    NBLOCKS = 32 if c.shape[0] >=256 else 1
    BLOCK1 = c.shape[0] // NBLOCKS
    BLOCK1_SUB = BLOCK1 if BLOCK1 < 64 else 64
    BLOCK2 = c.shape[3] if c.shape[3] < 64 else 64

    value = torch.empty_strided((c.shape[0], c.shape[1], c.shape[2], c.shape[3]),
                                (c.stride()[0], c.stride()[1], c.stride()[2], c.stride()[3]), dtype=torch.float32).npu()
    
    triton_foo[NBLOCKS, 1, 1](a, b, c, value, BLOCK1, BLOCK1_SUB, BLOCK2,
                   c.shape[0], c.shape[1], c.shape[2], c.shape[3],
                   a.stride()[0], a.stride()[1], a.stride()[2], a.stride()[3],
                   c.stride()[0], c.stride()[1], c.stride()[2], c.stride()[3],)
    return value

def foo(a, b, c):
    y = a + b
    y = c + y.permute(2, 1, 0, 3)
    return y


def test_permute_handwritten():

    Z, Y, X, R = (1, 12, 4096, 8)
    a = torch.randn((Z, Y, X, R), dtype=torch.float32).npu()
    b = torch.randn((Z, Y, X, R), dtype=torch.float32).npu()
    c = torch.randn((X, Y, Z, R), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    print(r[0, 0, 0:8, 0:8])
    print(r1[0, 0, 0:8, 0:8])
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)