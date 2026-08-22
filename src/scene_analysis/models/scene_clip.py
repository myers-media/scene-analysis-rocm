from __future__ import annotations

from PIL import Image

from ..device import ComputeDevice, synchronize
from ..types import SceneTag

DEFAULT_MODEL = "openai/clip-vit-base-patch32"

SCENE_PROMPTS = [
    "an indoor living room",
    "a kitchen interior",
    "a bedroom",
    "an office interior",
    "a restaurant or cafe",
    "a retail store",
    "a warehouse or factory",
    "a construction site",
    "a city street",
    "a highway or road",
    "a parking lot",
    "a crowded public plaza",
    "a sports field or stadium",
    "a beach",
    "a forest",
    "a mountain landscape",
    "a lake or river",
    "a city skyline",
    "an aerial landscape",
    "a night scene",
    "an industrial facility",
    "a hospital or clinic",
    "a classroom or lecture hall",
    "a park",
    "an airport or transit hub",
]


class SceneClassifier:
    """Zero-shot CLIP scene tagging. Runs on HIP via torch.cuda on ROCm builds."""

    def __init__(
        self,
        device: ComputeDevice,
        model_name: str = DEFAULT_MODEL,
        prompts: list[str] | None = None,
    ):
        self.device = device
        self.model_name = model_name
        self.prompts = prompts or list(SCENE_PROMPTS)
        self._model = None
        self._processor = None
        self._text_features = None

    def _load(self):
        if self._model is not None and self._text_features is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name)
            if self.device.torch_device is not None:
                self._model = self._model.to(self.device.torch_device)
            self._model.eval()
        text_inputs = self._processor(text=self.prompts, return_tensors="pt", padding=True)
        if self.device.torch_device is not None:
            text_inputs = {k: v.to(self.device.torch_device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            feats = embedding_from_clip_output(self._model.get_text_features(**text_inputs))
            self._text_features = feats / feats.norm(dim=-1, keepdim=True)

    def predict(self, image: Image.Image, top_k: int = 5) -> list[SceneTag]:
        self._load()
        import torch

        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        if self.device.torch_device is not None:
            inputs = {k: v.to(self.device.torch_device) for k, v in inputs.items()}
        with torch.inference_mode():
            image_features = embedding_from_clip_output(self._model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)
            scores = logits[0].detach().float().cpu()
        synchronize(self.device)
        top_k = max(1, min(top_k, len(self.prompts)))
        values, indices = scores.topk(top_k)
        return [
            SceneTag(label=_clean_prompt(self.prompts[int(i)]), score=float(v))
            for v, i in zip(values.tolist(), indices.tolist())
        ]

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._text_features = None


def embedding_from_clip_output(output):
    """Transformers 4 returns a tensor; 5+ returns BaseModelOutputWithPooling."""
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _clean_prompt(prompt: str) -> str:
    text = prompt
    for prefix in ("an ", "a "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text
