from __future__ import annotations

from pathlib import Path

from scene_analysis.device import classify_backend, read_rocm_version


def test_classify_rocm_when_hip_and_gpu_visible():
    assert classify_backend(hip_version="6.3.42131", cuda_available=True) == "rocm"


def test_classify_cuda_without_hip():
    assert classify_backend(hip_version=None, cuda_available=True) == "cuda"


def test_classify_cpu_when_rocm_build_has_no_gpu():
    assert classify_backend(hip_version="6.3.0", cuda_available=False) == "cpu"


def test_classify_cpu_default():
    assert classify_backend(hip_version=None, cuda_available=False) == "cpu"


def test_read_rocm_version_from_info_file(tmp_path: Path):
    info = tmp_path / ".info"
    info.mkdir()
    (info / "version").write_text("6.3.3-47\n", encoding="utf-8")
    assert read_rocm_version(tmp_path) == "6.3.3-47"


def test_read_rocm_version_missing(tmp_path: Path):
    assert read_rocm_version(tmp_path) is None
