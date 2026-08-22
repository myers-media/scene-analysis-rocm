from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..types import SceneAnalysis
from .grok import DEFAULT_MODEL as GROK_MODEL
from .grok import enrich_with_grok
from .lmstudio import DEFAULT_BASE_URL as LM_STUDIO_URL
from .lmstudio import DEFAULT_MODEL as LM_STUDIO_MODEL
from .lmstudio import enrich_with_lmstudio

PROVIDERS = ("lmstudio", "grok")


@dataclass
class NarrativeConfig:
    provider: str = "lmstudio"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    include_image: bool = False

    def normalized_provider(self) -> str:
        name = (self.provider or "lmstudio").strip().lower()
        if name in {"local", "lm-studio", "lm studio", "lms"}:
            return "lmstudio"
        if name in {"xai", "spacexai", "grok-4.6"}:
            return "grok"
        if name not in PROVIDERS:
            raise ValueError(f"Unknown narrative provider {self.provider!r}. Use lmstudio or grok.")
        return name


def enrich_scene(
    image: Image.Image,
    analysis: SceneAnalysis,
    config: NarrativeConfig | None = None,
) -> str:
    cfg = config or NarrativeConfig()
    provider = cfg.normalized_provider()
    if provider == "grok":
        return enrich_with_grok(
            image,
            analysis,
            api_key=cfg.api_key,
            model=cfg.model or GROK_MODEL,
        )
    return enrich_with_lmstudio(
        image,
        analysis,
        model=cfg.model or LM_STUDIO_MODEL,
        base_url=cfg.base_url or LM_STUDIO_URL,
        api_key=cfg.api_key,
        include_image=cfg.include_image,
    )
