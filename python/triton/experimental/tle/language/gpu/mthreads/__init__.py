# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from . import buffer, common, copy, pipe, warp_specialize, wgmma
from .types import MusaDotOperandEncoding, MusaSqmmaEncoding, MusaWmmaEncoding

# Public capability markers for clients that need to retain a safe fallback
# when running against an older FlagTree wheel.  Consumer-sync version 1
# covers physical-CTA barrier initialization, cyclic multi-epoch named-reader
# rendezvous, and mixed-warp rectangular shared-memory slice operands.
MTHREADS_GEMM_COMPILER_OPTIONS_VERSION = 2
MTHREADS_TLE_CONSUMER_SYNC_VERSION = 1

__all__ = [
    "buffer",
    "common",
    "copy",
    "pipe",
    "warp_specialize",
    "wgmma",
    "MusaDotOperandEncoding",
    "MusaSqmmaEncoding",
    "MusaWmmaEncoding",
    "MTHREADS_GEMM_COMPILER_OPTIONS_VERSION",
    "MTHREADS_TLE_CONSUMER_SYNC_VERSION",
]
