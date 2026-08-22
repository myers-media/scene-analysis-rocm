from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from .composition import analyze_composition
from .device import ComputeDevice, detect_device, empty_cache
from .types import SceneAnalysis, Timing


@dataclass
class TaskSet:
    detect: bool = True
    scene: bool = True
    caption: bool = False
    depth: bool = False
    composition: bool = True
    grok: bool = False

    @classmethod
    def from_names(cls, names: list[str] | tuple[str, ...]) -> "TaskSet":
        flags = {n.strip().lower() for n in names}
        return cls(
            detect="detect" in flags or "all" in flags,
            scene="scene" in flags or "all" in flags,
            caption="caption" in flags or "all" in flags,
            depth="depth" in flags or "all" in flags,
            composition="composition" in flags or "all" in flags,
            grok="grok" in flags or "narrative" in flags,
        )


@dataclass
class ScenePipeline:
    device: ComputeDevice = field(default_factory=detect_device)
    detector_name: str = "yolo11n.pt"
    keep_models_loaded: bool = False
    detector: object | None = None
    classifier: object | None = None
    captioner: object | None = None
    depth_estimator: object | None = None

    def _time(self, task: str, fn):
        start = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - start
        return value, Timing(task=task, seconds=round(elapsed, 3), backend=self.device.backend)

    def _maybe_unload(self, worker) -> None:
        if self.keep_models_loaded:
            return
        unload = getattr(worker, "unload", None)
        if callable(unload):
            unload()
        empty_cache(self.device)

    def _get_detector(self):
        if self.detector is None:
            from .models.detection import ObjectDetector

            self.detector = ObjectDetector(self.device, self.detector_name)
        return self.detector

    def _get_classifier(self):
        if self.classifier is None:
            from .models.scene_clip import SceneClassifier

            self.classifier = SceneClassifier(self.device)
        return self.classifier

    def _get_captioner(self):
        if self.captioner is None:
            from .models.caption import Captioner

            self.captioner = Captioner(self.device)
        return self.captioner

    def _get_depth(self):
        if self.depth_estimator is None:
            from .models.depth import DepthEstimator

            self.depth_estimator = DepthEstimator(self.device)
        return self.depth_estimator

    def analyze_image(
        self,
        image: Image.Image,
        tasks: TaskSet | None = None,
        *,
        conf: float = 0.25,
        grok_api_key: str | None = None,
        narrative=None,
        source: str = "image",
    ) -> SceneAnalysis:
        tasks = tasks or TaskSet()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        result = SceneAnalysis(
            width=rgb.width,
            height=rgb.height,
            device=self.device.snapshot(),
            source=source,
        )

        if self.device.backend == "unavailable" and (tasks.detect or tasks.scene or tasks.caption or tasks.depth):
            result.warnings.append(
                "PyTorch is not installed, so GPU vision models are disabled. "
                "Composition analysis still runs."
            )

        def run_task(enabled: bool, name: str, fn) -> None:
            if not enabled:
                return
            try:
                value, timing = self._time(name, fn)
                result.timings.append(timing)
                return value
            except Exception as exc:
                result.warnings.append(f"{name} failed: {exc}")
                return None

        if tasks.composition:
            composition = run_task(True, "composition", lambda: analyze_composition(rgb))
            if composition is not None:
                result.composition = composition

        if tasks.detect and self.device.backend != "unavailable":
            detector = self._get_detector()
            detections = run_task(True, "detect", lambda: detector.predict(rgb, conf=conf))
            if detections is not None:
                result.detections = detections
            self._maybe_unload(detector)

        if tasks.scene and self.device.backend != "unavailable":
            classifier = self._get_classifier()
            tags = run_task(True, "scene", lambda: classifier.predict(rgb))
            if tags is not None:
                result.scene_tags = tags
            self._maybe_unload(classifier)

        if tasks.caption and self.device.backend != "unavailable":
            captioner = self._get_captioner()
            caption = run_task(True, "caption", lambda: captioner.predict(rgb))
            if caption:
                result.caption = caption
            self._maybe_unload(captioner)

        if tasks.depth and self.device.backend != "unavailable":
            depth_est = self._get_depth()
            depth = run_task(True, "depth", lambda: depth_est.predict(rgb))
            if depth is not None:
                result.depth = depth
            self._maybe_unload(depth_est)

        if tasks.grok:
            try:
                from .llm.narrative import NarrativeConfig, enrich_scene

                cfg = narrative or NarrativeConfig()
                if grok_api_key and cfg.normalized_provider() == "grok":
                    cfg.api_key = cfg.api_key or grok_api_key
                label = cfg.normalized_provider()
                text, timing = self._time(
                    label,
                    lambda: enrich_scene(rgb, result, cfg),
                )
                result.timings.append(timing)
                result.grok_narrative = text
            except Exception as exc:
                result.warnings.append(f"narrative failed: {exc}")

        return result
