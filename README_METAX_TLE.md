# MetaX TLE (mctle) build fixes

This branch is `0.6.1+metax3.6` plus the fixes needed to actually run TLE
(`triton.experimental.tle`, shared-memory buffers and `local_ptr`) on a MetaX
C550. Each fix is its own commit:

| commit | what | why |
|---|---|---|
| `[Metax][TLE] Metax TLE support local pointer (#971)` | cherry-pick from `main` | the tag predates it; without it `mctle.local_pointers` lowers to an `llvm.bitcast` across address spaces and fails verification |
| `Keep shared buffers reached through mctle.local_pointers alive` | `third_party/metax/lib/Analysis/Alias.cpp` | metax's own alias analysis lacks the `local_pointers` case `lib/Analysis/Alias.cpp` has under `__TLE__`, so the shared-memory allocator reuses live TLE buffers for scratch and other buffers -- **silently wrong results** |
| `Pass __MCTLE__ to TableGen as well` | `cmake/FlagTreeOptions.cmake` | `__MCTLE__` was defined for C++ only; `TritonOps.td` guards #971's `tt.atomic_rmw` / `tt.atomic_cas` pointer constraint with it, so every `tt.atomic_rmw` on a shared pointer failed the verifier with "ptr type matches value type" |

## Build

Same as the metax CI. `BUILD_MCTLE` defaults to `ON` for the metax backend
(`cmake/FlagTreeOptions.cmake`), so nothing extra is needed:

```bash
export FLAGTREE_BACKEND=metax
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

LLVM and `metaxTritonPlugin.so` (v0.6.2) are downloaded prebuilt by setup, as
usual. The plugin cannot be built from source here: `plugin/mctle/dialect/lib`
publishes only `IR`, so `FLAGTREE_PLUGIN=1` fails at configure.

Install into a separate environment rather than over a working Triton.

## Check

```python
from triton._C import libtriton
import triton.backends.metax.compiler as c
print(hasattr(libtriton.ir.builder, "make_swizzled_shared_encoding_attr"),  # True
      c.enable_mctle)                                                      # True
```

## Still open (not fixed here)

These are in the prebuilt plugin or missing bindings; the FlagGems-vllm
operator works around them (see
`src/flaggems_vllm/runtime/backend/_metax/fused/top_k_per_row_tle.py`):

- **Vector width of unmasked shared loads.** The plugin's `__MCTLE__` block in
  `LoadOpConversion` widens `vec` from pointer alignment alone, not clamped by
  elements per thread, and then asserts
  `wordNElems * nWords * numVecs == numElems` whenever a thread holds fewer than
  4 elements (e.g. 512 lanes on 8 warps of 64). Aborts the process.
- **Masked shared atomics on replicated 1-D layouts** (e.g. a `[512]` tensor on
  4 or 2 warps of 64 lanes) write to wrong byte offsets. 1 element per thread
  and `[512, 4]` tiles are correct.
- **Missing bindings on metax:** `create_exclusive_cumsum` (`tle.cumsum`) and
  `get_memdesc_type` (passing a TLE buffer into a `@jit` function).

## Measured effect

FlagGems-vllm `top_k_per_row` on a C550, kernel time relative to vLLM's own
kernels: decode 0.964 -> 1.760, prefill 0.640 -> 1.194 when the TLE path is
enabled on a build with these fixes.
