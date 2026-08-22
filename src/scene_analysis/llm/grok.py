from __future__ import annotations

import os

from PIL import Image

from ..types import SceneAnalysis
from .common import image_data_url, narrative_prompt, structured_context

DEFAULT_MODEL = "grok-4.6"
XAI_BASE_URL = "https://api.x.ai/v1"

# Back-compat aliases used by tests.
_image_data_url = image_data_url
_structured_context = structured_context


def enrich_with_grok(
    image: Image.Image,
    analysis: SceneAnalysis,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask Grok (SpaceXAI / xAI) for a narrative that uses local ROCm vision results."""
    key = api_key or os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set. Add it to the environment or .env.")

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=XAI_BASE_URL)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_data_url(image), "detail": "high"},
                    {"type": "input_text", "text": narrative_prompt(analysis)},
                ],
            }
        ],
    )
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    return str(response)
