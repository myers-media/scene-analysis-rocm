from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def make_demo_image(width: int = 960, height: int = 540) -> Image.Image:
    """Synthetic outdoor street-like scene so the UI works without an upload."""
    image = Image.new("RGB", (width, height), (92, 148, 196))
    draw = ImageDraw.Draw(image)
    sx = width / 960.0
    sy = height / 540.0

    def box(x1: float, y1: float, x2: float, y2: float) -> list[int]:
        return [
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        ]

    for y in range(height // 2):
        t = y / max(1, height // 2)
        color = (int(92 + 40 * t), int(148 + 20 * t), int(196 - 30 * t))
        draw.line([(0, y), (width, y)], fill=color)
    draw.rectangle([0, int(height * 0.55), width, height], fill=(72, 78, 70))
    draw.rectangle([0, int(height * 0.68), width, int(height * 0.84)], fill=(48, 48, 50))
    mid = int(height * 0.75)
    draw.rectangle([0, mid - 4, width, mid + 4], fill=(214, 186, 64))
    draw.ellipse(box(780, 40, 880, 140), fill=(255, 214, 92))
    draw.rectangle(box(80, 160, 280, 313), fill=(176, 122, 90))
    for y in range(180, 280, 36):
        for x in range(100, 260, 40):
            draw.rectangle(box(x, y, x + 22, y + 22), fill=(210, 230, 240))
    draw.rectangle(box(340, 300, 365, 400), fill=(92, 64, 40))
    draw.ellipse(box(300, 220, 410, 330), fill=(46, 120, 62))
    draw.ellipse(box(620, 290, 655, 325), fill=(232, 196, 160))
    draw.rectangle(box(628, 325, 648, 410), fill=(40, 70, 140))
    draw.rectangle(box(612, 340, 628, 390), fill=(40, 70, 140))
    draw.rectangle(box(648, 340, 664, 390), fill=(40, 70, 140))
    try:
        draw.rounded_rectangle(box(720, 360, 900, 430), radius=max(4, int(18 * sx)), fill=(200, 48, 48))
    except Exception:
        draw.rectangle(box(720, 360, 900, 430), fill=(200, 48, 48))
    draw.ellipse(box(745, 410, 785, 450), fill=(30, 30, 30))
    draw.ellipse(box(835, 410, 875, 450), fill=(30, 30, 30))
    draw.rectangle(box(790, 372, 860, 400), fill=(180, 220, 230))
    try:
        font = ImageFont.truetype("arial.ttf", max(10, int(22 * sy)))
    except OSError:
        font = ImageFont.load_default()
    draw.text((int(16 * sx), height - int(36 * sy)), "Synthetic demo scene", fill=(240, 240, 240), font=font)
    return image
