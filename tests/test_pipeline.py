from __future__ import annotations

from PIL import Image

from scene_analysis.device import ComputeDevice
from scene_analysis.pipeline import ScenePipeline, TaskSet
from scene_analysis.types import BoundingBox, SceneTag


class FakeDetector:
    def predict(self, image, conf=0.25):
        return [BoundingBox(1, 2, 40, 50, "car", 0.88)]

    def unload(self):
        self.unloaded = True


class FakeClassifier:
    def predict(self, image, top_k=5):
        return [SceneTag("city street", 0.71), SceneTag("parking lot", 0.12)]

    def unload(self):
        pass


def cpu_device() -> ComputeDevice:
    return ComputeDevice(
        backend="cpu",
        name="CPU",
        index=-1,
        torch_device="cpu",
        memory_total_gb=None,
        hip_version=None,
        pytorch_version="test",
        rocm_system=None,
        notes=(),
    )


def test_taskset_from_names():
    tasks = TaskSet.from_names(["detect", "caption"])
    assert tasks.detect and tasks.caption
    assert not tasks.scene and not tasks.depth and not tasks.grok
    all_tasks = TaskSet.from_names(["all"])
    assert all_tasks.detect and all_tasks.depth and all_tasks.caption


def test_pipeline_with_injected_models():
    pipeline = ScenePipeline(
        device=cpu_device(),
        detector=FakeDetector(),
        classifier=FakeClassifier(),
        keep_models_loaded=True,
    )
    image = Image.new("RGB", (64, 48), (30, 80, 160))
    result = pipeline.analyze_image(image, TaskSet(detect=True, scene=True, composition=True))
    assert result.width == 64
    assert result.detections[0].label == "car"
    assert result.scene_tags[0].label == "city street"
    assert result.composition is not None
    task_names = {t.task for t in result.timings}
    assert {"detect", "scene", "composition"} <= task_names
    payload = result.to_dict()
    assert payload["detections"][0]["label"] == "car"


def test_pipeline_skips_vision_without_pytorch():
    device = ComputeDevice(
        backend="unavailable",
        name="PyTorch not installed",
        index=-1,
        torch_device=None,
        memory_total_gb=None,
        hip_version=None,
        pytorch_version=None,
        rocm_system=None,
        notes=(),
    )
    pipeline = ScenePipeline(device=device)
    image = Image.new("RGB", (32, 32), (10, 10, 10))
    result = pipeline.analyze_image(image, TaskSet(detect=True, scene=True, composition=True))
    assert result.detections == []
    assert result.composition is not None
    assert any("PyTorch is not installed" in w for w in result.warnings)
