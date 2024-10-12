import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ..base_model import BaseModel


class Vision2SeqModel(BaseModel):
    def __init__(self, model_id: str, **kwargs):
        self.model_name = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, **kwargs
        ).to(self.device, torch.bfloat16)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()

    def preprocess(self, image: str | Image.Image, prompt: str = None):
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                raise ValueError("Input string must be an image URL")
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be either an image URL or a PIL Image")

        return self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

    def predict(self, preprocessed_input, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=torch.bfloat16
        ):
            return self.model.generate(**preprocessed_input, **generate_kwargs)

    def postprocess(self, prediction):
        output = self.processor.batch_decode(prediction, skip_special_tokens=True)[0]
        return output.replace("\n", "").strip()

    def inference(self, image, prompt, **generate_kwargs):
        preprocessed_input = self.preprocess(image, prompt)
        prediction = self.predict(preprocessed_input, **generate_kwargs)
        return self.postprocess(prediction)
