"""A non-positive ``coalesced_width`` must be rejected, not crash the compiler.

``PlanLoopPartition`` computed ``vector_size % coalesced_width`` before checking
the width. A zero width is a representable value (``T.copy(..., coalesced_width=0)``)
and produced a C++ modulo-by-zero — an uncatchable SIGFPE that kills the compiler
worker. It must instead raise a clear, catchable error.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
import tvm


def _kernel(coalesced_width: int):
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128) as bx:
            s = T.alloc_shared((128, 128), "float32")
            T.copy(A, s, coalesced_width=coalesced_width)
            T.copy(s, B, coalesced_width=coalesced_width)

    return tvm.IRModule.from_expr(
        main.with_attr({"global_symbol": "main", "target": tvm.target.Target("cuda")}))


def test_zero_coalesced_width_raises():
    mod = _kernel(0)
    with tvm.target.Target("cuda"):
        with pytest.raises(Exception, match="coalesced_width must be a positive"):
            tilelang.transform.LowerTileOp()(tilelang.transform.LayoutInference()(mod))


if __name__ == "__main__":
    tilelang.testing.main()
