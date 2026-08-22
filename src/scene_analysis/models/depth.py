from __future__ import annotations

import numpy as np
from PIL import Image

from ..device import ComputeDevice
from ..types import DepthMap
from ..viz import colorize_depth, png_bytes

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, device: ComputeDevice, model_name: str = DEFAULT_MODEL):
        self.device = device
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline

        kwargs = {"task": "depth-estimation", "model": self.model_name}
        if self.device.backend in {"rocm", "cuda"}:
            kwargs["device"] = self.device.transformers_device
        self._pipe = pipeline(**kwargs)

    def predict(self, image: Image.Image) -> DepthMap:
        self._load()
        result = self._pipe(image.convert("RGB"))
        depth = result.get("predicted_depth")
        if hasattr(depth, "detach"):
            depth_np = depth.detach().float().cpu().numpy()
        else:
            depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()
        preview = colorize_depth(depth_np)
        if preview.size != image.size:
            preview = preview.resize(image.size, Image.BILINEAR)
        return DepthMap(
            width=int(depth_np.shape[1] if depth_np.ndim == 2 else image.width),
            height=int(depth_np.shape[0] if depth_np.ndim == 2 else image.height),
            min_value=float(np.min(depth_np)),
            max_value=float(np.max(depth_np)),
            mean_value=float(np.mean(depth_np)),
            preview_png=png_bytes(preview),
        )

    def unload(self) -> None:
        self._pipe = None
