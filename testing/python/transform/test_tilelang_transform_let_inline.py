from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm.tirx.stmt_functor import post_order_visit


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LetInline()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), True)


def test_let_binding():
    @T.prim_func
    def before(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    factor = T.float32(2.0)
                    value = A[i, j] * factor
                    B[i, j] = value

    @T.prim_func
    def expected(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    B[i, j] = A[i, j] * T.float32(2.0)

    _check(before, expected)


def test_parallel_scope():
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.sblock("parallel"):
                value = T.float32(1.0)
                A[i] = value

    @T.prim_func
    def expected(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.sblock("parallel"):
                A[i] = T.float32(1.0)

    _check(before, expected)


def test_effectful_binding_is_not_duplicated():
    @T.prim_func
    def before(A: T.Tensor((2,), T.uint32)):
        value = T.rng_rand()
        A[0] = value
        A[1] = value

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.LetInline()(mod)
    calls = []
    post_order_visit(
        mod["main"].body,
        lambda node: calls.append(node) if isinstance(node, tvm.tirx.Call) and node.op.same_as(T.rng_rand().op) else None,
    )

    assert len(calls) == 1


if __name__ == "__main__":
    tilelang.testing.main()
