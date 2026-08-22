from __future__ import annotations

from PIL import Image

from ..device import ComputeDevice
from ..types import BoundingBox

DEFAULT_MODEL = "yolo11n.pt"


class ObjectDetector:
    """Ultralytics YOLO on the resolved ROCm/CUDA/CPU torch device."""

    def __init__(self, device: ComputeDevice, model_name: str = DEFAULT_MODEL):
        self.device = device
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        return self._model

    def predict(self, image: Image.Image, conf: float = 0.25) -> list[BoundingBox]:
        model = self._load()
        results = model.predict(
            image,
            conf=conf,
            device=self.device.ultralytics_device,
            verbose=False,
        )
        detections: list[BoundingBox] = []
        if not results:
            return detections
        result = results[0]
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()
        for box, score, cls_id in zip(xyxy, confs, clss):
            label = names.get(int(cls_id), str(int(cls_id)))
            detections.append(
                BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    label=str(label),
                    confidence=float(score),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def unload(self) -> None:
        self._model = None
