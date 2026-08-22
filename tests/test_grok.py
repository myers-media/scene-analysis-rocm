from __future__ import annotations

from PIL import Image

from scene_analysis.llm.grok import _image_data_url, _structured_context
from scene_analysis.types import BoundingBox, SceneAnalysis, SceneTag


def test_data_url_is_jpeg():
    image = Image.new("RGB", (16, 16), (12, 34, 56))
    url = _image_data_url(image)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 40


def test_structured_context_includes_detections():
    analysis = SceneAnalysis(
        width=10,
        height=10,
        caption="a red car on a street",
        detections=[BoundingBox(0, 0, 5, 5, "car", 0.9)],
        scene_tags=[SceneTag("city street", 0.4)],
    )
    blob = _structured_context(analysis)
    assert "car" in blob
    assert "city street" in blob
    assert "a red car" in blob
