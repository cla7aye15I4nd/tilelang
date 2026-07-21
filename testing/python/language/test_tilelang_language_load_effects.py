import pytest

import tilelang
from tilelang import tvm


@pytest.mark.parametrize(
    "op_name",
    [
        "tl.ds_read_tr16_b64",
        "tl.ds_read_tr8_b64",
        "tl.__ldg",
        "tl.ldg32",
        "tl.ldg64",
        "tl.ldg128",
    ],
)
def test_gpu_load_intrinsics_are_marked_as_state_reads(op_name):
    effect = tvm.ir.Op.get(op_name).get_attr("TCallEffectKind")

    assert int(effect) == tvm.tirx.CallEffectKind.ReadState.value


if __name__ == "__main__":
    tilelang.testing.main()
