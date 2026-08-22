from __future__ import annotations

from PIL import Image

from scene_analysis.types import BoundingBox, SceneTag
from scene_analysis.viz import colorize_depth, draw_detections, png_bytes, turbo_colormap
import numpy as np


def test_draw_detections_keeps_size():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    boxes = [BoundingBox(10, 10, 80, 70, "person", 0.91)]
    tags = [SceneTag("city street", 0.6)]
    out = draw_detections(image, boxes, tags)
    assert out.size == image.size
    assert out.mode == "RGB"
    assert png_bytes(out)[:8] == b"\x89PNG\r\n\x1a\n"


def test_colorize_depth_shape():
    depth = np.linspace(0, 1, 50 * 40, dtype=np.float32).reshape(40, 50)
    preview = colorize_depth(depth)
    assert preview.size == (50, 40)
    cmap = turbo_colormap(np.array([[0.0, 1.0]], dtype=np.float32))
    assert cmap.shape == (1, 2, 3)
