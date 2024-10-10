import requests
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from ..base_model import BaseModel


class BLIP2(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.processor = Blip2Processor.from_pretrained(self.model_name, **kwargs)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        ).to(self.device, torch.bfloat16)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()

    def preprocess(self, image: str | Image.Image, prompt: str = None):
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                raise ValueError("Input string must be an image URL for BLIP2")
        else:
            raise ValueError(
                "Input must be either an image URL or a PIL Image for BLIP2"
            )

        return self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

    def predict(self, processed_data, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=torch.bfloat16
        ):
            return self.model.generate(**processed_data, **generate_kwargs)

    def postprocess(self, prediction):
        return self.processor.batch_decode(prediction, skip_special_tokens=True)[0]
