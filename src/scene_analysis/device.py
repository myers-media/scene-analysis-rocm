from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComputeDevice:
    """Resolved inference target. On ROCm, PyTorch still uses the `cuda` device API."""

    backend: str  # rocm | cuda | directml | cpu | unavailable
    name: str
    index: int
    torch_device: Any
    memory_total_gb: float | None
    hip_version: str | None
    pytorch_version: str | None
    rocm_system: str | None
    notes: tuple[str, ...] = ()

    @property
    def is_gpu(self) -> bool:
        return self.backend in {"rocm", "cuda", "directml"}

    @property
    def ultralytics_device(self) -> int | str:
        if self.backend in {"rocm", "cuda"}:
            return self.index
        return "cpu"

    @property
    def transformers_device(self) -> int:
        if self.backend in {"rocm", "cuda"}:
            return self.index
        return -1

    def snapshot(self):
        from .types import DeviceSnapshot

        return DeviceSnapshot(
            backend=self.backend,
            name=self.name,
            index=self.index,
            memory_total_gb=self.memory_total_gb,
            hip_version=self.hip_version,
            pytorch_version=self.pytorch_version,
            rocm_system=self.rocm_system,
        )


def classify_backend(*, hip_version: str | None, cuda_available: bool) -> str:
    """Map PyTorch build flags onto a user-facing accelerator name."""
    if hip_version:
        return "rocm" if cuda_available else "cpu"
    if cuda_available:
        return "cuda"
    return "cpu"


def read_rocm_version(root: str | Path | None = None) -> str | None:
    """Read the installed ROCm stack version from /opt/rocm or ROC_PATH."""
    candidates = []
    if root:
        candidates.append(Path(root))
    env = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    if env:
        candidates.append(Path(env))
    candidates.extend([Path("/opt/rocm"), Path("C:/Program Files/AMD/ROCm")])
    for base in candidates:
        for rel in (".info/version", "share/doc/rocm/VERSION", "bin/.info/version"):
            path = base / rel
            try:
                if path.is_file():
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        return text.splitlines()[0].strip()
            except OSError:
                continue
    return None


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or proc.stderr or "").strip()
    return out or None


def _try_directml():
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
        name = torch_directml.device_name(0)
        return device, str(name)
    except Exception:
        return None


def detect_device(prefer_gpu: bool = True) -> ComputeDevice:
    """Pick the best available torch device. ROCm builds expose HIP as torch.cuda."""
    notes: list[str] = []
    rocm_system = read_rocm_version()

    try:
        import torch
    except ImportError:
        notes.append("PyTorch is not installed. Install a ROCm or CPU wheel — see README.")
        return ComputeDevice(
            backend="unavailable",
            name="PyTorch not installed",
            index=-1,
            torch_device=None,
            memory_total_gb=None,
            hip_version=None,
            pytorch_version=None,
            rocm_system=rocm_system,
            notes=tuple(notes),
        )

    hip_version = getattr(torch.version, "hip", None)
    pytorch_version = torch.__version__
    cuda_available = bool(torch.cuda.is_available()) if prefer_gpu else False
    backend = classify_backend(hip_version=hip_version, cuda_available=cuda_available)

    if backend in {"rocm", "cuda"} and prefer_gpu:
        index = 0
        try:
            name = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            memory = float(getattr(props, "total_memory", 0)) / (1024**3)
        except Exception as exc:  # pragma: no cover - hardware specific
            notes.append(f"GPU probe failed: {exc}")
            name = "AMD GPU" if backend == "rocm" else "NVIDIA GPU"
            memory = None
        if backend == "rocm":
            notes.append("ROCm is using the PyTorch HIP compatibility layer (torch.cuda).")
        return ComputeDevice(
            backend=backend,
            name=name,
            index=index,
            torch_device=torch.device("cuda", index),
            memory_total_gb=round(memory, 2) if memory else None,
            hip_version=str(hip_version) if hip_version else None,
            pytorch_version=pytorch_version,
            rocm_system=rocm_system,
            notes=tuple(notes),
        )

    dml = _try_directml() if prefer_gpu else None
    if dml is not None:
        device, name = dml
        notes.append("Using DirectML. Native ROCm wheels are Linux/WSL2 only.")
        return ComputeDevice(
            backend="directml",
            name=name,
            index=0,
            torch_device=device,
            memory_total_gb=None,
            hip_version=str(hip_version) if hip_version else None,
            pytorch_version=pytorch_version,
            rocm_system=rocm_system,
            notes=tuple(notes),
        )

    if hip_version and not cuda_available:
        notes.append(
            "This is a ROCm PyTorch build, but no AMD GPU is visible. "
            "Check HIP_VISIBLE_DEVICES, amdgpu driver, and `rocminfo`."
        )
    elif platform.system() == "Windows":
        notes.append(
            "Official ROCm PyTorch wheels target Linux. On Windows use WSL2 + ROCm, "
            "or run on CPU. DirectML is used automatically if torch-directml is installed."
        )
    else:
        notes.append("Running on CPU. Install a ROCm PyTorch wheel to use AMD GPUs.")

    return ComputeDevice(
        backend="cpu",
        name="CPU",
        index=-1,
        torch_device=torch.device("cpu"),
        memory_total_gb=None,
        hip_version=str(hip_version) if hip_version else None,
        pytorch_version=pytorch_version,
        rocm_system=rocm_system,
        notes=tuple(notes),
    )


def gpu_memory_allocated_gb(device: ComputeDevice) -> float | None:
    if device.backend not in {"rocm", "cuda"} or device.torch_device is None:
        return None
    try:
        import torch

        return round(torch.cuda.memory_allocated(device.index) / (1024**3), 3)
    except Exception:
        return None


def synchronize(device: ComputeDevice) -> None:
    if device.backend not in {"rocm", "cuda"}:
        return
    try:
        import torch

        torch.cuda.synchronize(device.index)
    except Exception:
        return


def empty_cache(device: ComputeDevice) -> None:
    if device.backend not in {"rocm", "cuda"}:
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        return


def matmul_gflops(device: ComputeDevice, size: int = 2048, repeats: int = 5) -> float | None:
    """Rough GEMM throughput probe used by the UI diagnostics panel."""
    if device.torch_device is None:
        return None
    try:
        import time

        import torch
    except ImportError:
        return None

    try:
        a = torch.randn(size, size, device=device.torch_device, dtype=torch.float32)
        b = torch.randn(size, size, device=device.torch_device, dtype=torch.float32)
        torch.matmul(a, b)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            torch.matmul(a, b)
        synchronize(device)
        elapsed = (time.perf_counter() - start) / repeats
        # 2 * N^3 FLOPs for GEMM
        flops = 2.0 * (size**3)
        return round((flops / elapsed) / 1e9, 1)
    except Exception:
        return None


def probe_rocm() -> dict[str, Any]:
    """Collect a JSON-serializable snapshot of ROCm / PyTorch state."""
    device = detect_device()
    rocminfo = _run(["rocminfo"])
    rocm_smi = _run(["rocm-smi"])
    hipinfo = _run(["hipinfo"]) or _run(["hipconfig", "--full"])
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": {
            "backend": device.backend,
            "name": device.name,
            "index": device.index,
            "memory_total_gb": device.memory_total_gb,
            "hip_version": device.hip_version,
            "pytorch_version": device.pytorch_version,
            "rocm_system": device.rocm_system,
            "notes": list(device.notes),
        },
        "env": {
            key: os.environ.get(key)
            for key in (
                "ROCM_PATH",
                "HIP_PATH",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "HSA_OVERRIDE_GFX_VERSION",
                "HSA_ENABLE_SDMA",
                "PYTORCH_ROCM_ARCH",
            )
            if os.environ.get(key)
        },
        "rocminfo_head": (rocminfo.splitlines()[:40] if rocminfo else None),
        "rocm_smi": rocm_smi,
        "hipinfo_head": (hipinfo.splitlines()[:40] if hipinfo else None),
    }


def main() -> None:
    import json

    print(json.dumps(probe_rocm(), indent=2))


if __name__ == "__main__":
    main()
