from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from scene_analysis.web_camera import image_from_data_url


def test_image_from_data_url_roundtrip():
    buf = BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buf, format="JPEG")
    data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    image = image_from_data_url(data)
    assert image is not None
    assert image.size[0] == 4
    assert image.mode == "RGB"


def test_image_from_data_url_rejects_junk():
    assert image_from_data_url(None) is None  # type: ignore[arg-type]
    assert image_from_data_url("not-an-image") is None
