import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[3] / "cmake" / "find_pip_cuda.py"
SPEC = importlib.util.spec_from_file_location("find_pip_cuda", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
find_pip_cuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(find_pip_cuda)


def test_cuda_stub_does_not_follow_fixed_source_symlink(tmp_path):
    cu_dir = tmp_path / "cu"
    stubs_dir = cu_dir / "lib" / "stubs"
    stubs_dir.mkdir(parents=True)

    victim = tmp_path / "victim.c"
    victim.write_text("do not overwrite\n")
    (stubs_dir / "_stub.c").symlink_to(victim)

    find_pip_cuda._ensure_cuda_stub(cu_dir)

    assert victim.read_text() == "do not overwrite\n"
    assert (stubs_dir / "libcuda.so").is_file()
    assert not list(stubs_dir.glob("tilelang_cuda_stub_*"))
