"""The pure helpers behind assert_runtime.py's build-variant checks.

The checks themselves run inside the images at build time (that is the point
of them). What is tested here is the logic that decides pass or fail, each
case PROVEN able to fail: a check that cannot reject the wrong build is an
assumption, not a check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from assert_runtime import (  # noqa: E402
    CT2_GPU_MARKER,
    DRIVER_PROVIDED,
    TORCH_BUILDS,
    VARIANTS,
    build_of,
    check_build,
    file_contains,
    main,
    parse_ldd_missing,
    unresolved_libraries,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "build"),
    [("2.15.0+cpu", "cpu"), ("2.15.0+cu132", "cu132"), ("2.15.0+rocm7.14", "rocm7.14"), ("2.15.0", "")],
)
def test_build_of(version: str, build: str) -> None:
    assert build_of(version) == build


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "want", "ok"),
    [
        ("2.15.0+cpu", "cpu", True),
        ("2.15.0+cu132", "cu", True),
        ("2.11.0+cu130", "cu", True),
        ("2.15.0+rocm7.14", "rocm", True),
        # The failure #4's class of bug hides behind: right version, wrong build.
        ("2.15.0", "cpu", False),
        ("2.15.0+cu132", "cpu", False),
        ("2.15.0+cpu", "cu", False),
        ("2.15.0+cpu", "rocm", False),
        ("2.15.0+rocm7.14", "cu", False),
        # "cpu" must not match by substring of something else.
        ("2.15.0+xpu", "cpu", False),
    ],
)
def test_check_build(version: str, want: str, ok: bool) -> None:
    failures: list[str] = []
    check_build(failures, "torch", version, want)
    assert (failures == []) is ok
    if not ok:
        assert version in failures[0] and f"+{want}" in failures[0]


@pytest.mark.unit
def test_every_gpu_variant_has_its_builds_and_ct2_marker() -> None:
    for variant in ("cuda", "rocm"):
        assert variant in VARIANTS
        assert set(TORCH_BUILDS[variant]) == {"torch", "torchaudio", "torchcodec"}
        assert CT2_GPU_MARKER[variant]
    # PyTorch publishes no ROCm torchcodec: the ROCm image ships the CPU one.
    assert TORCH_BUILDS["rocm"]["torchcodec"] == "cpu"
    assert TORCH_BUILDS["full"] == {"torch": "cpu", "torchaudio": "cpu", "torchcodec": "cpu"}
    # torchaudio 2.11 has no cu132 build and its cu130 one refuses torch cu132.
    assert TORCH_BUILDS["cuda"]["torchaudio"] == "cpu"


@pytest.mark.unit
def test_file_contains_finds_a_marker_split_across_chunks(tmp_path: Path) -> None:
    lib = tmp_path / "libctranslate2.so"
    marker = b"libcublas.so.12"
    # Put the marker straddling the 16-byte chunk boundary.
    lib.write_bytes(b"x" * 10 + marker + b"y" * 40)
    assert file_contains(lib, marker, chunk=16)
    assert not file_contains(lib, b"libamdhip64.so", chunk=16)


@pytest.mark.unit
def test_file_contains_on_an_empty_file(tmp_path: Path) -> None:
    lib = tmp_path / "empty.so"
    lib.write_bytes(b"")
    assert not file_contains(lib, b"libcublas.so.12")


@pytest.mark.unit
def test_unknown_variant_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["--variant", "metal"])


LDD_OUTPUT = """\
\tlinux-vdso.so.1 (0x00007ffd1a3f2000)
\tlibcublasLt.so.13 => /app/.venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13 (0x00007f0)
\tlibcudnn.so.9 => not found
\tlibcuda.so.1 => not found
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1)
"""


@pytest.mark.unit
def test_parse_ldd_missing_reports_only_unresolved_sonames() -> None:
    assert parse_ldd_missing(LDD_OUTPUT) == {"libcudnn.so.9", "libcuda.so.1"}


@pytest.mark.unit
def test_only_the_driver_may_be_missing() -> None:
    # The CUDA image gate: libcuda.so.1 comes from the host at run time, but a
    # missing cuDNN is a broken image and must not be excused with it.
    assert parse_ldd_missing(LDD_OUTPUT) - DRIVER_PROVIDED == {"libcudnn.so.9"}
    resolved = LDD_OUTPUT.replace("libcudnn.so.9 => not found", "")
    assert parse_ldd_missing(resolved) - DRIVER_PROVIDED == set()


@pytest.mark.unit
def test_unresolved_libraries_fails_on_a_missing_file(tmp_path: Path) -> None:
    failures: list[str] = []
    assert unresolved_libraries(failures, tmp_path / "nope.so") is None
    assert failures and "does not exist" in failures[0]
