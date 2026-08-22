"""ROCm-accelerated scene analysis."""

from .camera import live_taskset, parse_camera_source
from .device import ComputeDevice, detect_device, probe_rocm
from .pipeline import ScenePipeline, TaskSet
from .types import SceneAnalysis

__version__ = "1.0.0"
__all__ = [
    "ComputeDevice",
    "SceneAnalysis",
    "ScenePipeline",
    "TaskSet",
    "live_taskset",
    "parse_camera_source",
    "detect_device",
    "probe_rocm",
    "__version__",
]
