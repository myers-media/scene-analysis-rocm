from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import BoundingBox, SceneTag

PALETTE = [
    (237, 28, 36),
    (0, 168, 204),
    (255, 184, 28),
    (46, 196, 126),
    (155, 89, 182),
    (255, 107, 53),
    (88, 166, 255),
    (244, 162, 97),
]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _color_for(label: str) -> tuple[int, int, int]:
    return PALETTE[abs(hash(label)) % len(PALETTE)]


def draw_detections(
    image: Image.Image,
    detections: list[BoundingBox],
    tags: list[SceneTag] | None = None,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(max(14, canvas.width // 80))
    for det in detections:
        color = _color_for(det.label)
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
        caption = f"{det.label} {det.confidence:.0%}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 4], fill=color + (220,))
        draw.text((x1 + 4, ty + 1), caption, fill=(255, 255, 255, 255), font=font)

    if tags:
        chip_x, chip_y = 12, 12
        for tag in tags[:4]:
            text = f"{tag.label} {tag.score:.0%}"
            bbox = draw.textbbox((chip_x, chip_y), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rounded_rectangle(
                [chip_x, chip_y, chip_x + tw + 14, chip_y + th + 8],
                radius=8,
                fill=(20, 22, 28, 210),
                outline=(237, 28, 36, 255),
            )
            draw.text((chip_x + 7, chip_y + 3), text, fill=(244, 241, 236, 255), font=font)
            chip_y += th + 16

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Approximate Google Turbo without a matplotlib dependency."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(0.135 + 2.9 * v - 2.3 * (v - 0.5) ** 2, 0, 1)
    g = np.clip(0.05 + 2.4 * v - 3.6 * (v - 0.55) ** 2, 0, 1)
    b = np.clip(1.1 - 1.9 * v + 1.4 * (v - 0.35) ** 2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> Image.Image:
    finite = np.asarray(depth, dtype=np.float32)
    finite = np.nan_to_num(finite, nan=0.0)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-6:
        norm = np.zeros_like(finite)
    else:
        norm = (finite - lo) / (hi - lo)
    rgb = turbo_colormap(norm)
    return Image.fromarray(rgb, mode="RGB")


def png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
