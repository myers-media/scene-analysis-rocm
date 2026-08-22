from __future__ import annotations

from PIL import Image

from ..device import ComputeDevice, synchronize


DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


class Captioner:
    def __init__(self, device: ComputeDevice, model_name: str = DEFAULT_MODEL):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        if self.device.torch_device is not None:
            self._model = self._model.to(self.device.torch_device)
        self._model.eval()

    def predict(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        self._load()
        import torch

        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        if self.device.torch_device is not None:
            inputs = {k: v.to(self.device.torch_device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        synchronize(self.device)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
