# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = tl.cuda.transform.LowerSharedTmem()(mod)
    return mod


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tirx.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def _collect_tmem_blocks(stmt):
    blocks = []

    def visitor(node):
        if isinstance(node, tvm.tirx.SBlock) and any(buf.scope() == "shared.tmem" for buf in node.alloc_buffers):
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return blocks


def test_explicit_deallocate_tmem_suppresses_auto_dealloc():
    """Explicit T.deallocate_tmem on fallthrough suppresses auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            T.deallocate_tmem(C_tmem)

    mod = _apply(func)
    body = mod["main"].body
    assert len(_collect_calls(body, "tl.ptx_init_tensor_memory")) == 1
    assert len(_collect_calls(body, "tl.ptx_deallocate_tensor_memory")) == 1
    assert len(_collect_calls(body, "tl.deallocate_tmem")) == 0

    dealloc_call = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")[0]
    assert dealloc_call.args[1].value == 128


def test_explicit_deallocate_only_suppresses_matching_buffer():
    """Only the explicitly-deallocated buffer skips auto-dealloc; others keep it."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            A_tmem = T.alloc_tmem([128, 128], T.float32)
            B_tmem = T.alloc_tmem([128, 64], T.float32)
            T.deallocate_tmem(A_tmem)

    mod = _apply(func)
    body = mod["main"].body

    dealloc_calls = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    # A_tmem: 1 explicit (auto suppressed); B_tmem: 1 auto = 2 total
    assert len(dealloc_calls) == 2

    dealloc_num_cols = sorted(call.args[1].value for call in dealloc_calls)
    assert dealloc_num_cols == [64, 128]


def test_dealloc_before_thread_return_keeps_auto_dealloc():
    """Dealloc on non-fallthrough path (before thread_return) does NOT suppress auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            tx = T.get_thread_binding()

            if tx < 32:
                T.deallocate_tmem(C_tmem)
                T.thread_return()

    mod = _apply(func)
    body = mod["main"].body

    dealloc_calls = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    # 1 explicit (non-fallthrough) + 1 auto (block end) = 2
    assert len(dealloc_calls) == 2
    assert [call.args[1].value for call in dealloc_calls] == [128, 128]


def test_sibling_blocks_lower_only_their_own_tmem_allocations():
    """TMEM allocations from an earlier sibling must not be lowered twice."""

    @T.prim_func
    def first_func():
        with T.Kernel(1, threads=128):
            first_tmem = T.alloc_tmem([128, 128], T.float32)  # noqa: F841

    @T.prim_func
    def second_func():
        with T.Kernel(1, threads=128):
            second_tmem = T.alloc_tmem([128, 64], T.float32)  # noqa: F841

    def get_tmem_block(func):
        mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
        mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        blocks = _collect_tmem_blocks(mod["main"].body)
        assert len(blocks) == 1
        return blocks[0]

    thread_var = tvm.tirx.Var("threadIdx.x", "int32")
    thread_iter = tvm.tirx.IterVar(
        tvm.ir.Range(0, 128), thread_var, tvm.tirx.IterVar.ThreadIndex, "threadIdx.x"
    )
    body = tvm.tirx.AttrStmt(
        thread_iter,
        "thread_extent",
        128,
        tvm.tirx.SeqStmt([get_tmem_block(first_func), get_tmem_block(second_func)]),
    )
    func = tvm.tirx.PrimFunc([], body).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.cuda.transform.LowerSharedTmem()(mod)

    init_calls = _collect_calls(mod["main"].body, "tl.ptx_init_tensor_memory")
    dealloc_calls = _collect_calls(mod["main"].body, "tl.ptx_deallocate_tensor_memory")
    assert sorted(call.args[1].value for call in init_calls) == [64, 128]
    assert sorted(call.args[1].value for call in dealloc_calls) == [64, 128]


if __name__ == "__main__":
    test_explicit_deallocate_tmem_suppresses_auto_dealloc()
    test_explicit_deallocate_only_suppresses_matching_buffer()
    test_dealloc_before_thread_return_keeps_auto_dealloc()
