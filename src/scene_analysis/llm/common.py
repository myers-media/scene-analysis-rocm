from __future__ import annotations

import base64
import json
from io import BytesIO

from PIL import Image

from ..types import SceneAnalysis

NARRATIVE_PROMPT = (
    "You are a scene-analysis specialist. Using the image (if provided) and the local "
    "AMD ROCm vision results below, write a concise scene report with these sections:\n"
    "1. Setting and likely time of day\n"
    "2. Key people, vehicles, and objects, including spatial relationships\n"
    "3. Lighting, weather, and mood\n"
    "4. Notable risks, anomalies, or things a human operator should notice\n"
    "Keep it under 250 words. Do not invent objects that contradict the detections.\n"
)


def image_data_url(image: Image.Image, max_side: int = 768, quality: int = 72) -> str:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def structured_context(analysis: SceneAnalysis) -> str:
    payload = {
        "caption": analysis.caption,
        "scene_tags": [{"label": t.label, "score": round(t.score, 3)} for t in analysis.scene_tags],
        "detections": [
            {
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "box": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
            }
            for d in analysis.detections[:40]
        ],
        "label_counts": analysis.label_counts(),
    }
    if analysis.composition:
        payload["composition"] = {
            "brightness": analysis.composition.brightness,
            "contrast": analysis.composition.contrast,
            "saturation": analysis.composition.saturation,
            "is_blurry": analysis.composition.is_blurry,
        }
    return json.dumps(payload, indent=2)


def narrative_prompt(analysis: SceneAnalysis) -> str:
    return f"{NARRATIVE_PROMPT}\nLocal ROCm vision results:\n{structured_context(analysis)}"
