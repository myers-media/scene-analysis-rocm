from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

_FRONTEND = Path(__file__).resolve().parent / "web_camera_frontend"
_component = None


def image_from_data_url(data: str) -> Image.Image | None:
    if not data or not isinstance(data, str) or "," not in data:
        return None
    try:
        _header, b64 = data.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def capture_browser_camera(
    key: str = "browser_camera",
    *,
    live: bool = False,
    interval_ms: int = 300,
    scan_token: int = 0,
) -> Image.Image | None:
    """Frames from the *browser* machine, with an explicit camera-device picker."""
    global _component
    import streamlit.components.v1 as components

    if _component is None:
        _component = components.declare_component("browser_camera", path=str(_FRONTEND))
    data = _component(
        live=bool(live),
        interval_ms=int(interval_ms),
        scan_token=int(scan_token),
        key=key,
        default=None,
    )
    return image_from_data_url(data)
